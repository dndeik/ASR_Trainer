import os
import time
import random
import numpy as np
import torch
import torchaudio
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
from typing import Optional
import werpy
from torch.utils.tensorboard import SummaryWriter

from utils.decoding import rnnt_greedy_decode_batch, ctc_greedy_decode_batch
from feature_extractor.feature_extractor import FeaturesExractor
from models.common_modules.regularization import SpecAugment, TokenDropout


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config, model, tokenizer, blank_token_id, optimizer, scheduler,
                 train_dataloader, validation_dataloader, train_sampler, args):
        set_seed(time.time_ns() % 10 ** 6)

        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.blank_token_id = blank_token_id
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accum_step = config["trainer"]["grad_accum"]

        self.use_amp = config["trainer"].get("train_with_amp", False)
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        # Для BF16 GradScaler не нужен, так как динамический диапазон совпадает с FP32
        self.scaler = None

        self.feature_extractor = FeaturesExractor(**config["FFT"], n_mel=config["model"]["n_mel"]).to(self.device)
        self.feature_extractor.freeze()
        self.spec_aug = SpecAugment(0.2, 0.15, 30, 15, p=0.85, adaptive_time_mask=True).to(self.device)
        self.token_dropout = TokenDropout(0.1)

        self.ctc_loss = torch.nn.modules.CTCLoss(blank=self.blank_token_id, zero_infinity=True).to(self.device)
        self.rnnt_loss = torchaudio.functional.rnnt_loss
        self.ctc_weight = 0.3

        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.save_n_last_checkpoints = self.trainer_config.get('save_n_last_checkpoints', -1)
        self.validation_interval_steps = self.trainer_config['validation_interval_steps']
        self.last_val_step = 0
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']

        # experiment config
        self.exp_config = config['experiment']
        self.exp_root_folder = self.exp_config['experiments_path']
        self.exp_name = self.exp_config['exp_name']
        self.resume = self.exp_config['resume']
        self.resume_from_folder = self.exp_config['resume_from_folder']

        if not self.resume:
            self.exp_path = os.path.join(self.exp_root_folder,
                                         '_'.join(self.exp_name.split()) + '_' + datetime.now().strftime(
                                             "%Y-%m-%d-%Hh%Mm"))
            self._save_model_info()
        elif self.resume_from_folder:
            self.exp_path = self.resume_from_folder
        else:
            raise RuntimeError("Set way to resume train")

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        # save the config
        with open(os.path.join(self.exp_path, 'config.toml'), 'w') as f:
            toml.dump(config, f)

        self.writer = SummaryWriter(self.log_path)
        # If you want to use clearml logger you should install package and use init
        if config["logger"]["log_to_clearml"]:
            from clearml import Logger
            self.clearml_logger = Logger.current_logger()
            self._save_clearml_info_to_file()

        self.start_epoch = 1
        self.global_step = 0
        self.best_score = float('inf')
        self.state_dict_best = None

        self.aug_start_epoch = config["trainer"]["epoch_when_enable_aug"]

        if self.resume:
            self._resume_checkpoint()

    def _save_clearml_info_to_file(self):
        from clearml import Task
        task_id = Task.current_task().id
        with open(os.path.join(self.exp_path, 'clearml_info.txt'), "w", encoding="utf-8") as f:
            f.write(self.exp_name + "\n")
            f.write(task_id)

    def _save_model_info(self):
        os.makedirs(self.exp_path, exist_ok=True)
        with open(os.path.join(self.exp_path, 'model_config.txt'), "w", encoding="utf-8") as f:
            f.write(str(self.model))

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, step: int, score: Optional[float]):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        epoch = (step * self.accum_step) // len(self.train_dataloader)

        state_dict = {
            'step': step,
            'epoch': epoch,
            'best_score': self.best_score,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model': model_dict
        }

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(step).zfill(10)}.tar'))

        if self.save_n_last_checkpoints > 0:
            all_checkpoints = []
            for fname in os.listdir(self.checkpoint_path):
                if fname.startswith('model_') and fname.endswith('.tar') and fname != 'best_model.tar':
                    all_checkpoints.append(fname)

            def extract_epoch(filename):
                return int(filename.split('_')[1].split('.')[0])

            all_checkpoints.sort(key=extract_epoch)

            if len(all_checkpoints) > self.save_n_last_checkpoints:
                to_delete = all_checkpoints[:-self.save_n_last_checkpoints]
                for old_name in to_delete:
                    old_path = os.path.join(self.checkpoint_path, old_name)
                    try:
                        os.remove(old_path)
                    except OSError as e:
                        print(f'Failed to delete {old_path}: {e}')

        if score is not None and score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

            torch.save(self.state_dict_best,
                       os.path.join(self.checkpoint_path,
                                    'best_model.tar'.format(str(self.state_dict_best['step']).zfill(10))))
            print(f'New best models saved on step {step}')

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location, weights_only=False)

        self.global_step = checkpoint['step'] + 1
        self.start_epoch = (self.global_step * self.accum_step) // len(self.train_dataloader)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_score = checkpoint.get('best_score', float('inf'))

        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'], strict=True)

    def _calculate_wer(self, esti: torch.Tensor, target: torch.Tensor, type: str = "rnnt"):
        blank_token = self.blank_token_id

        if type == "ctc":
            esti = esti.transpose(0, 1)
            esti = torch.argmax(esti, dim=-1).detach().int()
            batch_decoded = ctc_greedy_decode_batch(esti, blank=blank_token)

        elif type == "rnnt":
            batch_decoded = rnnt_greedy_decode_batch(esti, blank=blank_token)

        batch_decoded = [[x for x in sublist if x < blank_token] for sublist in batch_decoded]
        esti_list = [self.tokenizer.decode(batch_el) for batch_el in batch_decoded]

        target = target.detach().int()
        target_batch_decoded = ctc_greedy_decode_batch(target, blank=blank_token)
        target_batch_decoded = [[x for x in sublist if x < blank_token] for sublist in target_batch_decoded]
        target_list = [self.tokenizer.decode(batch_el) for batch_el in target_batch_decoded]

        norm_esti_list = werpy.normalize(esti_list)
        norm_target_list = werpy.normalize(target_list)

        norm_esti_list = [el.replace("¿", "").replace("¡", "").strip() for el in norm_esti_list]
        norm_target_list = [el.replace("¿", "").replace("¡", "").strip() for el in norm_target_list]

        wer = werpy.wer(norm_target_list, norm_esti_list)
        wer = wer if wer is not None else 1.

        return wer, esti_list, target_list

    def _model_invoke(self, audio, audio_len, target, target_len):
        target = torch.cat(
            [torch.full((target.shape[0], 1), fill_value=self.blank_token_id, device=target.device), target],
            dim=-1)
        ctc_out, rnnt_out, corrected_audio_len = self.model(audio, audio_len, target, target_len + 1)
        corrected_audio_len = corrected_audio_len.long()
        ctc_out = ctc_out.transpose(0, 1)
        return ctc_out, rnnt_out, corrected_audio_len

    def _calculate_loss(self, ctc_logits, rnnt_logits, targets, x_lengths, target_lengths):
        ctc_logits = ctc_logits.float()
        rnnt_logits = rnnt_logits.float()

        targets = targets.to(torch.int32)
        x_lengths = x_lengths.to(torch.int32)
        target_lengths = target_lengths.to(torch.int32)

        ctc_weight = 0.75
        rnnt_weight = 1

        if ctc_logits.shape[0] > x_lengths.max():
            print("WARNING: logits more then x_lengths max")
            ctc_logits = ctc_logits[:x_lengths.max(), ...]

        ctc_log_probs = torch.nn.functional.log_softmax(ctc_logits, dim=-1)
        loss_ctc = self.ctc_loss(ctc_log_probs, targets, x_lengths, target_lengths)

        # torchaudio's CUDA kernel can overflow int32 indexing on very large grids.
        batch_size, time_size, target_size, vocab_size = rnnt_logits.shape
        grid_size = batch_size * time_size * target_size * vocab_size
        if grid_size < 2_000_000_000:
            loss_rnnt = self.rnnt_loss(
                rnnt_logits, targets, x_lengths, target_lengths, blank=self.blank_token_id
            )
        else:
            max_batch = max(1, 2_000_000_000 // (time_size * target_size * vocab_size))
            chunk_losses = []
            for start in range(0, batch_size, max_batch):
                end = min(start + max_batch, batch_size)
                chunk_x_lengths = x_lengths[start:end]
                chunk_target_lengths = target_lengths[start:end]
                chunk_time = int(chunk_x_lengths.max())
                chunk_target = int(chunk_target_lengths.max())
                chunk_losses.append(self.rnnt_loss(
                    rnnt_logits[start:end, :chunk_time, :chunk_target + 1].contiguous(),
                    targets[start:end, :chunk_target].contiguous(),
                    chunk_x_lengths,
                    chunk_target_lengths,
                    blank=self.blank_token_id,
                    reduction="sum",
                ))
            loss_rnnt = torch.stack(chunk_losses).sum() / batch_size

        lambda_logit_weight = 1e-4
        logit_norm_penalty = rnnt_logits.pow(2).mean() * lambda_logit_weight

        loss = ctc_weight * loss_ctc + rnnt_weight * loss_rnnt + logit_norm_penalty

        return {
            'total_loss': loss,
            'ctc_loss': loss_ctc,
            'rnnt_loss': loss_rnnt,
            'logit_norm_loss': logit_norm_penalty,
        }

    @staticmethod
    def _compute_joiner_penalties(
            rnnt_logits,
            blank_token_id,
            lambda_margin,
            margin_threshold,
            lambda_blank,
            blank_threshold,
            lambda_logit_norm,
    ):
        """
        Пенальти на сырые логиты joiner'а.
        """
        rnnt_logits = rnnt_logits.float()

        # 1. Logit Margin Penalty
        sorted_logits, _ = torch.sort(rnnt_logits, dim=-1, descending=True)
        margin = sorted_logits[..., 0] - sorted_logits[..., 1]  # [B, T, U+1]
        margin_penalty = torch.nn.functional.relu(margin - margin_threshold).pow(2).mean()

        # 2. Blank Confidence Penalty
        probs = torch.softmax(rnnt_logits, dim=-1)
        p_blank = probs[..., blank_token_id]  # [B, T, U+1]
        blank_penalty = torch.nn.functional.relu(p_blank - blank_threshold).pow(2).mean()

        # 3. Logit Norm Penalty
        logit_norm_penalty = rnnt_logits.pow(2).mean()

        total = (
                lambda_margin * margin_penalty
                + lambda_blank * blank_penalty
                + lambda_logit_norm * logit_norm_penalty
        )

        return {
            'total': total,
            'margin': margin_penalty,
            'blank': blank_penalty,
            'logit_norm': logit_norm_penalty,
        }

    @torch.no_grad()
    def _validate(self):
        total_val_losses = {}
        total_ctc_wer = 0
        total_rnnt_wer = 0
        validation_bar = tqdm(self.validation_dataloader, ncols=160)
        for step, (audio, audio_len, target, target_len) in enumerate(validation_bar, 1):
            audio = audio.to(self.device)
            audio_len = audio_len.to(self.device).long()
            features, audio_len = self.feature_extractor(audio, audio_len)

            target = target.to(self.device).long()
            target_len = target_len.to(self.device).long()

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    ctc_out, rnnt_out, corrected_audio_len = self._model_invoke(features, audio_len, target, target_len)
                    loss_dict = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)
            else:
                ctc_out, rnnt_out, corrected_audio_len = self._model_invoke(features, audio_len, target, target_len)
                loss_dict = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)

            for k, v in loss_dict.items():
                total_val_losses[k] = total_val_losses.get(k, 0) + v.item()

            batch_rnnt_wer, esti_rnnt_text, target_text = self._calculate_wer(rnnt_out, target, type="rnnt")
            total_rnnt_wer += batch_rnnt_wer

            batch_ctc_wer, esti_ctc_text, target_text = self._calculate_wer(ctc_out, target, type="ctc")
            total_ctc_wer += batch_ctc_wer

            validation_bar.desc = f'validate[{self.global_step}][{datetime.now().strftime("%Y-%m-%d-%H:%M")}]'
            avg_val_loss = total_val_losses.get('total_loss', 0) / step
            validation_bar.postfix = f'valid_loss={avg_val_loss:.2f}, valid_ctc_wer={total_ctc_wer / step:.2f}, valid_rnnt_wer={total_rnnt_wer / step:.2f}'

            validation_bar.write("")
            validation_bar.write(f"REF: {target_text[0]}")
            validation_bar.write(f"EST: {esti_rnnt_text[0]}")

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            for k, v in total_val_losses.items():
                self.writer.add_scalar(f'Epoch_{k}/val', v / step, self.global_step)
            self.writer.add_scalar('Epoch_WER/val(ctc)', total_ctc_wer / step, self.global_step)
            self.writer.add_scalar('Epoch_WER/val(rnnt)', total_rnnt_wer / step, self.global_step)

        return total_rnnt_wer / step

    def _validate_and_save(self, step):
        self._set_eval_mode()
        score = self._validate()

        if self.rank == 0:
            self._save_checkpoint(step, score)

    def train(self):
        if self.resume:
            self._resume_checkpoint()

        total_losses = {}
        accum_losses = {}

        total_wer = 0
        accum_wer = 0

        steps_from_last_val = 0

        for epoch in range(self.start_epoch, self.epochs):
            if epoch >= self.aug_start_epoch:
                self.train_dataloader.dataset.augs_enable = True
                print("Augs enabled")
            else:
                self.train_dataloader.dataset.augs_enable = False
                print("Augs disabled")

            set_seed(time.time_ns() % 10 ** 6)
            self._set_train_mode()

            self.resume = False

            train_bar = tqdm(self.train_dataloader, total=len(self.train_dataloader), ncols=160)
            self.optimizer.zero_grad()

            for step, (audio, audio_len, target, target_len) in enumerate(train_bar, 1):
                audio = audio.to(self.device)
                audio_len = audio_len.to(self.device).long()
                features, audio_len = self.feature_extractor(audio, audio_len)
                features = self.token_dropout(features)
                features = self.spec_aug(features)

                target = target.to(self.device).long()
                target_len = target_len.to(self.device).long()

                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        ctc_out, rnnt_out, corrected_audio_len = self._model_invoke(features, audio_len, target,
                                                                                    target_len)
                        loss_dict = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)
                else:
                    ctc_out, rnnt_out, corrected_audio_len = self._model_invoke(features, audio_len, target, target_len)
                    loss_dict = self._calculate_loss(ctc_out, rnnt_out, target, corrected_audio_len, target_len)

                loss = loss_dict['total_loss']

                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN/Inf loss detected, skipping step.")
                    self.optimizer.zero_grad()
                    continue

                for k, v in loss_dict.items():
                    val = v.item()
                    total_losses[k] = total_losses.get(k, 0) + val
                    accum_losses[k] = accum_losses.get(k, 0) + val

                self._optimization_step(step, loss)

                wer_type = "ctc"
                batch_wer, esti_text, target_text = self._calculate_wer(ctc_out, target, type=wer_type)
                total_wer += batch_wer
                accum_wer += batch_wer

                steps_from_last_val += 1

                train_bar.desc = f'train[{epoch}/{self.epochs}][{datetime.now().strftime("%Y-%m-%d-%H:%M")}]'

                avg_loss = total_losses.get('total_loss', 0) / max(steps_from_last_val, 1)
                avg_wer = total_wer / max(steps_from_last_val, 1)
                train_bar.postfix = f'train_loss={avg_loss:.2f}, wer({wer_type})={avg_wer:.3f}, g_step={self.global_step}'

                log_step = self.accum_step
                if step % log_step == 0 and self.rank == 0:
                    train_bar.write("")
                    train_bar.write(f"REF: {target_text[0]}")
                    train_bar.write(f"EST: {esti_text[0]}")
                    self.writer.add_scalars('LR', {'lr': self.optimizer.param_groups[0]['lr']},
                                            self.global_step)
                    log_dict = {k: accum_losses.get(k, 0) / log_step for k in accum_losses}
                    self.writer.add_scalars('Monitoring', log_dict, self.global_step)
                    self.writer.add_scalars('WER', {'train_wer': accum_wer / log_step},
                                            self.global_step)
                    accum_losses = {}
                    accum_wer = 0

                if self.world_size > 1 and (self.device != torch.device("cpu")):
                    torch.cuda.synchronize(self.device)

                if self.rank == 0 and self.global_step % self.validation_interval_steps == 0 and self.global_step > 0 and self.global_step != self.last_val_step:
                    self._validate_and_save(self.global_step)
                    self.last_val_step = self.global_step
                    self._set_train_mode()

                    steps_in_cycle = self.validation_interval_steps * self.accum_step
                    for k, v in total_losses.items():
                        self.writer.add_scalar(f'Epoch_{k}/train', v / steps_in_cycle, self.global_step)
                    self.writer.add_scalar('Epoch_WER/train(ctc)', total_wer / steps_in_cycle, self.global_step)

                    total_losses = {}
                    total_wer = 0

                    steps_from_last_val = 0

        if self.rank == 0:
            torch.save(self.state_dict_best,
                       os.path.join(self.checkpoint_path,
                                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))

            print('------------Training for {} epochs has done!------------'.format(self.epochs))

    def _optimization_step(self, step, loss):
        loss = loss / self.accum_step

        # [ИЗМЕНЕНО ДЛЯ BF16] Прямой backward, без GradScaler
        loss.backward()

        if step % self.accum_step == 0:
            self._log_grad_norms(step)
            cur_gn = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1
            self.writer.add_scalars('Grad norm', {'grad_norm': cur_gn}, self.global_step)

    def _log_grad_norms(self, step):
        """Замер градиентов по группам параметров."""

        def grad_norm(params):
            grads = [p.grad.norm() for p in params if p.grad is not None]
            if not grads:
                return torch.tensor(0.0)
            return torch.norm(torch.stack(grads))

        enc_grad = grad_norm(self.model.encoder.parameters())
        rnnt_grad = grad_norm(self.model.rnnt_classifier.parameters())
        ctc_grad = grad_norm(self.model.ctc_classifier.parameters())
        dec_grad = grad_norm(self.model.rnnt_decoder.parameters())

        grad_dict = {
            'encoder': enc_grad,
            'rnnt_classifier': rnnt_grad,
            'ctc_classifier': ctc_grad,
            'rnnt_decoder': dec_grad,
        }
        self.writer.add_scalars('Grads', grad_dict, self.global_step)
