import os
import torch
import torchaudio
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
import werpy
from torch.utils.tensorboard import SummaryWriter

from datasets import BLANK_TOKEN_ID


class Trainer:
    def __init__(self, config, model, tokenizer, optimizer,
                 train_dataloader, validation_dataloader, train_sampler, args):
        
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size
    
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_with_amp = config["trainer"]["train_with_amp"]
        self.scaler = torch.amp.GradScaler() if self.train_with_amp else None

        self.ctc_loss = torch.nn.modules.CTCLoss(blank=BLANK_TOKEN_ID, zero_infinity=True).to(self.device)
        self.rnnt_loss = torchaudio.functional.rnnt_loss
        self.ctc_weight = 0.3

        self.time_factor = config["model"]["time_factor"]
        self.accum_step = config["trainer"]["grad_accum"]

        t0 = 1500 / int(len(train_dataloader) * config["trainer"]["epochs"] // self.accum_step)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config["optimizer"]["lr"],
                                                             div_factor=200, pct_start=t0, anneal_strategy='cos',
                                                             steps_per_epoch=len(train_dataloader) // self.accum_step,
                                                             epochs=config["trainer"]["epochs"])

        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.validation_interval = self.trainer_config['validation_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']
        self.resume_checkpoint_path = self.trainer_config['resume_checkpoint_path']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
            self._save_model_info()
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        # save the config
        with open(
                os.path.join(
                    self.exp_path, 'config.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:
            toml.dump(config, f)

        self.writer = SummaryWriter(self.log_path)
        # If you want to use clearml logger you should install package and use init
        if config["logger"]["log_to_clearml"]:
            from clearml import Logger
            self.clearml_logger = Logger.current_logger()

        self.start_epoch = 0
        self.best_score = 0

        if self.resume:
            self._resume_checkpoint()

        self.sr = config['listener']['listener_sr']

    def _save_model_info(self):
        os.makedirs(self.exp_path, exist_ok=True)
        with open(os.path.join(self.exp_path, 'model_config.txt'), "w", encoding="utf-8") as f:
            f.write(str(self.model))

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        if self.train_with_amp:
            state_dict = {'epoch': epoch,
                          'val_pesq': score,
                          'optimizer': self.optimizer.state_dict(),
                          'scheduler': self.scheduler.state_dict(),
                          'model': model_dict,
                          'scaler': self.scaler.state_dict()}
        else:
            state_dict = {'epoch': epoch,
                          'val_pesq': score,
                          'optimizer': self.optimizer.state_dict(),
                          'scheduler': self.scheduler.state_dict(),
                          'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

            torch.save(self.state_dict_best,
                       os.path.join(self.checkpoint_path,
                                    'best_model.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))
            print(f'New best models saved on {epoch} epoch.')

    def _resume_checkpoint(self):
        if self.resume_checkpoint_path != '':
            latest_checkpoints = self.resume_checkpoint_path
        else:
            latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        if self.train_with_amp:
            if 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            else:
                print("AMP is enabled but the checkpoint doesn't contain scaler state. New scaler will be initialized")

        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    @staticmethod
    def _rnnt_greedy_decode_batch(logits, blank=0):
        """
        logits: Tensor [B, T, U, V] - выход модели
        blank: индекс blank токена
        Returns:
        list из списков токенов для каждого примера в батче
        """

        B, T, U, V = logits.shape
        device = logits.device

        # Начальные позиции для каждого примера
        t_pos = torch.zeros(B, dtype=torch.int64, device=device)
        u_pos = torch.zeros(B, dtype=torch.int64, device=device)

        # Для каждого примера храним список предсказанных токенов
        results = [[] for _ in range(B)]

        # Флаг завершения для каждого примера
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        while (~finished).any():
            # Для всех не завершенных берем логиты на текущих позициях
            batch_indices = torch.arange(B, device=device)[~finished]

            t_indices = t_pos[~finished]
            u_indices = u_pos[~finished]

            # logits для текущих позиций: [active_batch, V]
            current_logits = logits[batch_indices, t_indices, u_indices, :]

            # Выбираем argmax токенов
            preds = torch.argmax(current_logits, dim=-1)

            # Обрабатываем выбор для каждого активного примера
            for i, b_idx in enumerate(batch_indices):
                if preds[i].item() == blank:
                    # Если blank — сдвигаем t
                    t_pos[b_idx] += 1
                    # Если вышли за предел T — считаем пример завершенным
                    if t_pos[b_idx] >= T:
                        finished[b_idx] = True
                else:
                    # Добавляем токен в результат
                    results[b_idx].append(preds[i].item())
                    u_pos[b_idx] += 1
                    # Если вышли за предел U — считаем пример завершенным
                    if u_pos[b_idx] >= U:
                        finished[b_idx] = True

            # Если все примеры закончили, выход из цикла
            if finished.all():
                break

        return results

    def _ctc_greedy_decode_batch(self, esti: torch.Tensor, blank: int):
        # 1. Создаем маску: токен != blank_token
        mask_not_blank = esti != blank  # shape [B, T]
        
        # 2. Создаем маску для удаления повторяющихся подряд токенов
        # Сдвинем по времени на 1 (влево) для сравнения с предыдущим токеном
        shifted = torch.zeros_like(esti)
        shifted[:, 1:] = esti[:, :-1]
        
        # Маска для выбора токенов, которые не равны предыдущему токену
        mask_no_repeat = esti != shifted  # shape [B, T]
        
        # Итоговая маска — токен не blank И не повторяется подряд
        mask = mask_not_blank & mask_no_repeat
        
        results = []
        for b in range(esti.size(0)):
            tokens = esti[b][mask[b]].cpu().tolist()
            results.append(tokens)
        
        return results

    def _calculate_wer(self, esti: torch.Tensor, target: torch.Tensor, type: str = "rnnt"):
        blank_token = BLANK_TOKEN_ID

        if type == "ctc":       
            esti = esti.transpose(0, 1)
            esti = torch.argmax(esti, dim=-1).detach().int()
            batch_decoded = self._ctc_greedy_decode_batch(esti, blank=blank_token)
            esti_list = [self.tokenizer.decode(batch_el) for batch_el in batch_decoded]


        elif type == "rnnt":
            batch_decoded = self._rnnt_greedy_decode_batch(esti, blank=blank_token)
            esti_list = [self.tokenizer.decode(batch_el) for batch_el in batch_decoded]

        # Targets processing
        target = target.detach().int()
        target_batch_decoded = self._ctc_greedy_decode_batch(target, blank=blank_token)
        target_list = [self.tokenizer.decode(batch_el) for batch_el in target_batch_decoded]

        norm_esti_list = werpy.normalize(esti_list)
        norm_target_list = werpy.normalize(target_list)

        wer = werpy.wer(norm_target_list, norm_esti_list)

        wer = wer if wer is not None else 1.

        return wer, esti_list, target_list


    def _model_invoke(self, audio, audio_len, target, target_len):
        target = torch.cat([torch.full((target.shape[0], 1), fill_value=BLANK_TOKEN_ID, device=target.device), target], dim=-1)
        ctc_out, rnnt_out = self.model(audio, audio_len, target, target_len+1)
        ctc_out = ctc_out.transpose(0, 1)
        return ctc_out, rnnt_out
    
    def _calculate_loss(self, ctc_logits, rnnt_logits, targets, x_lengths, target_lengths):
        targets = targets.to(torch.int32)
        x_lengths = x_lengths.to(torch.int32)
        target_lengths = target_lengths.to(torch.int32)

        if ctc_logits.shape[0] > x_lengths.max():
            print("WARNING: logits more then x_lengths max")
            ctc_logits = ctc_logits[:x_lengths.max(), ...]

        ctc_logits = torch.nn.functional.log_softmax(ctc_logits, dim=-1)
        rnnt_logits = torch.nn.functional.log_softmax(rnnt_logits, dim=-1)

        loss_rnnt = self.rnnt_loss(rnnt_logits, targets, x_lengths, target_lengths, blank=BLANK_TOKEN_ID)
        loss_ctc = self.ctc_loss(ctc_logits, targets, x_lengths, target_lengths)

        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_rnnt

        return loss, loss_ctc, loss_rnnt


    def _train_epoch(self, epoch):
        total_loss = 0
        accum_loss = 0

        total_ctc_loss = 0
        accum_ctc_loss = 0

        total_rnnt_loss = 0
        accum_rnnt_loss = 0

        total_wer = 0
        accum_wer = 0

        self.train_dataloader.dataset.explicit_shuffle()
        train_bar = tqdm(self.train_dataloader, ncols=160)
        self.optimizer.zero_grad()

        for step, (audio, audio_len, target, target_len) in enumerate(train_bar, 1):
            audio = audio.to(self.device)
            audio_len = audio_len.to(self.device).long() 
            corrected_audio_len = audio_len // self.time_factor # TODO: некрасиво
            corrected_audio_len = corrected_audio_len.long() 
            target = target.to(self.device).long() 
            target_len = target_len.to(self.device).long() 

            if self.train_with_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    ctc_out, rnnt_out = self._model_invoke(audio, corrected_audio_len, target, target_len)
                    loss, ctc_loss, rnnt_loss = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)
            else:
                ctc_out, rnnt_out = self._model_invoke(audio, corrected_audio_len, target, target_len)
                loss, ctc_loss, rnnt_loss = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)

            if not torch.isnan(loss):
                cur_loss = loss.item()
                total_loss += cur_loss
                accum_loss += cur_loss

                cur_ctc_loss = ctc_loss.item()
                total_ctc_loss += cur_ctc_loss
                accum_ctc_loss += cur_ctc_loss

                cur_rnnt_loss = rnnt_loss.item()
                total_rnnt_loss += cur_rnnt_loss
                accum_rnnt_loss += cur_rnnt_loss

                train_bar.desc = '   train[{}/{}][{}]'.format(
                    epoch, self.epochs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

                train_bar.postfix = 'train_loss={:.2f}, step={}'.format(total_loss / step, step // self.accum_step)

                # self.optimizer.zero_grad()

                if self.train_with_amp:
                    # self.optimizer.zero_grad()
                    loss = loss / self.accum_step
                    self.scaler.scale(loss).backward()
                    if step % self.accum_step == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                batch_wer, esti_text, target_text = self._calculate_wer(ctc_out, target, type="ctc")
                total_wer += batch_wer
                accum_wer += batch_wer

                log_step = self.accum_step
                if step % log_step == 0 and self.rank == 0:
                    print()
                    print("REF: ", target_text[0])
                    print("EST: ", esti_text[0])
                    real_step = (epoch * len(self.train_dataloader) + step) / self.accum_step
                    self.writer.add_scalars('LR', {'lr': self.optimizer.param_groups[0]['lr']},
                                            real_step)
                    self.writer.add_scalars('Monitoring', {'total_loss': accum_loss / log_step,
                                                           'ctc_loss': accum_ctc_loss / log_step,
                                                           'rnnt_loss': accum_rnnt_loss / log_step},
                                            real_step)
                    self.writer.add_scalars('WER', {'train_wer': accum_wer / log_step},
                                            real_step)
                    accum_loss = 0
                    accum_ctc_loss = 0
                    accum_rnnt_loss = 0
                    accum_wer = 0
            else:
                print("Nan loss")

            if self.world_size > 1 and (self.device != torch.device("cpu")):
                torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalar('Epoch_Total_Loss/train', total_loss / step, epoch)
            self.writer.add_scalar('Epoch_CTC_Loss/train', total_ctc_loss / step, epoch)
            self.writer.add_scalar('Epoch_RNNT_Loss/train', total_rnnt_loss / step, epoch)
            self.writer.add_scalar('Epoch_WER/train(ctc)', total_wer / step, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_ctc_wer = 0
        total_rnnt_wer = 0
        total_ctc_loss = 0
        total_rnnt_loss = 0
        validation_bar = tqdm(self.validation_dataloader, ncols=160)
        for step, (audio, audio_len, target, target_len) in enumerate(validation_bar, 1):
            audio = audio.to(self.device)
            audio_len = audio_len.to(self.device).long() 
            corrected_audio_len = audio_len // self.time_factor # TODO: некрасиво
            corrected_audio_len = corrected_audio_len.long() 
            target = target.to(self.device).long() 
            target_len = target_len.to(self.device).long() 

            ctc_out, rnnt_out = self._model_invoke(audio, corrected_audio_len, target, target_len)
            loss, ctc_loss, rnnt_loss = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_rnnt_loss += rnnt_loss.item()

            batch_rnnt_wer, esti_text, target_text = self._calculate_wer(rnnt_out, target, type="rnnt")
            total_rnnt_wer += batch_rnnt_wer

            batch_ctc_wer, esti_text, target_text = self._calculate_wer(ctc_out, target, type="ctc")
            total_ctc_wer += batch_ctc_wer

            validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            validation_bar.postfix = 'valid_loss={:.2f}, valid_ctc_wer={:.2f}, valid_rnnt_wer={:.2f}'.format(
                total_loss / step, total_ctc_wer / step, total_rnnt_wer / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalar('Epoch_Total_Loss/val', total_loss / step, epoch)
            self.writer.add_scalar('Epoch_CTC_Loss/val', total_ctc_loss / step, epoch)
            self.writer.add_scalar('Epoch_RNNT_Loss/val', total_rnnt_loss / step, epoch)
            self.writer.add_scalar('Epoch_WER/val(ctc)', total_ctc_wer / step, epoch)
            self.writer.add_scalar('Epoch_WER/val(rnnt)', total_rnnt_wer / step, epoch)

        return total_rnnt_wer / step


    def train(self):
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            if epoch % self.validation_interval == 0:
                valid_loss = self._validation_epoch(epoch)
            else:
                valid_loss = 0

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, valid_loss)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                       os.path.join(self.checkpoint_path,
                                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))

            print('------------Training for {} epochs has done!------------'.format(self.epochs))
