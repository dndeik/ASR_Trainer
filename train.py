import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import toml
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import sentencepiece as spm

from trainer import Trainer
from models.squeezeformer.model import ConformerHybrid, count_parameters
from datasets import MyDataset, DynamicBucketingSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# torch.backends.cudnn.deterministic =True
# torch.autograd.set_detect_anomaly(True)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def run(rank, config, args):
    args.rank = rank
    args.device = torch.device(rank)

    tokenizer = spm.SentencePieceProcessor(
        model_file=os.path.join(config['tokenizer']['tokenizer_path'], "tokenizer.model"))
    blank_token_id = tokenizer.vocab_size()
    multi_gpu = args.world_size > 1
    if multi_gpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("gloo", rank=rank, init_method="env://?use_libuv=False", world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    train_dataset = MyDataset(tokenizer=tokenizer, blank_token_id=blank_token_id,
                              **config['train_dataset'], is_train=True,
                              lang_switch_enable=config["augmentations"]["lang_switch_enable"])
    train_dataset.set_augmentations(**config["augmentations"])
    sampler_config = config["train_dataloader"]
    train_sampler = DynamicBucketingSampler(
        lengths=train_dataset.get_audio_lens(),
        token_lengths=train_dataset.get_text_lens(),
        max_seconds=sampler_config.get("max_seconds", 220),
        bucket_size=sampler_config.get("bucket_size", 8000),
        shuffle=True,
        drop_last=sampler_config.get("drop_last", True),
        max_tokens=sampler_config.get("max_tokens"),
        vocab_size=tokenizer.vocab_size(),
        frames_per_sec=(config["listener"]["listener_sr"]
                        / (config["FFT"]["hop_length"] * config["model"]["time_factor"])),
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        pin_memory=config['train_dataloader']['pin_memory'],
        num_workers=config['train_dataloader']['num_workers'],
        prefetch_factor=sampler_config.get("prefetch_factor", 1),
        shuffle=False,
        collate_fn=train_dataset.custom_collate_fn,
        worker_init_fn=worker_init_fn
    )

    validation_dataset = MyDataset(tokenizer=tokenizer, blank_token_id=blank_token_id,
                                   **config['validation_dataset'], is_train=False)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if multi_gpu else None
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        sampler=validation_sampler,
        **config['validation_dataloader'],
        shuffle=False,
        collate_fn=validation_dataset.custom_collate_fn,
        worker_init_fn=worker_init_fn
    )

    model = ConformerHybrid(num_vocab=tokenizer.vocab_size() + 1,
                            encoder_d_model=config['model']['encoder_d_model'],
                            predictor_d_model=config['model']['predictor_d_model'],
                            joiner_d_model=config['model']['joiner_d_model'],
                            n_mel=config['model']['n_mel'],
                            time_factor=config['model']['time_factor'],
                            chunk_size=config['model']['chunk_size'],
                            left_context_chunk_number=config['model']['left_context_chunk_number'],
                            right_context_chunk_number=config['model']['right_context_chunk_number'],
                            n_heads=config['model']['n_heads'],
                            n_groups=config['model']['n_groups'],
                            encoder_layer_num=config['model']['encoder_layer_num'],
                            predictor_layer_num=config['model']['predictor_layer_num'],
                            mamba_every_n_block=config['model']['mamba_every_n_block'],
                            dropout=0.2,
                            drop_path=0.15)

    if config["init_weights"]["checkpoint_path"]:
        ckpt = torch.load(config["init_weights"]["checkpoint_path"], map_location="cpu", weights_only=False)
        ckpt = ckpt["model"]

        model.load_state_dict(ckpt, strict=config["init_weights"]["strict"])
        print("Weight loaded")

    count_parameters(model)
    model.to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['optimizer']['lr'], weight_decay=0.01,
                                  betas=(0.9, 0.998))

    t0 = config["trainer"]["warm_up_steps"] / int(
        len(train_dataloader) * config["trainer"]["epochs"] // config["trainer"]["grad_accum"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4],
                                                    div_factor=200, pct_start=t0, anneal_strategy='cos',
                                                    steps_per_epoch=len(train_dataloader) // config["trainer"][
                                                        "grad_accum"],
                                                    epochs=config["trainer"]["epochs"])

    trainer = Trainer(config=config, model=model, tokenizer=tokenizer, blank_token_id=blank_token_id,
                      optimizer=optimizer, scheduler=scheduler,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
                      train_sampler=train_sampler, args=args)
    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='cfg_train.toml')

    args = parser.parse_args()
    config = json.loads(json.dumps(toml.load(args.config)))
    args.device = config['GPUs']['devices']
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))

    if config["logger"]["log_to_clearml"]:
        from clearml import Task

        if config["experiment"]["resume"]:
            with open(os.path.join(config["experiment"]["resume_from_folder"], "clearml_info.txt"), "r") as f:
                data = f.readlines()
            task_name = data[0]
            continue_task = data[1]
        else:
            task_name = config["experiment"]["exp_name"]
            continue_task = False
        task = Task.init(project_name='ASR', task_name=task_name, continue_last_task=continue_task)
        task.upload_artifact(name='config', artifact_object=config)

    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
