import os

import json
import toml
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import sentencepiece as spm

from trainer import Trainer
from models.conformer_plus_mamba.model import ConformerHybrid, count_parameters
from datasets import MyDataset, BLANK_TOKEN_ID, custom_collate_fn

# seed = 4956
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


# torch.backends.cudnn.deterministic =True
# torch.autograd.set_detect_anomaly(True)

def run(rank, config, args):
    args.rank = rank
    args.device = torch.device(rank)

    tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(config['tokenizer']['tokenizer_path'], "tokenizer.model"))

    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("gloo", rank=rank, init_method="env://?use_libuv=False", world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

        train_dataset = MyDataset(tokenizer=tokenizer, **config['train_dataset'], **config['FFT'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                       **config['train_dataloader'], shuffle=False,
                                                       collate_fn=custom_collate_fn)

        validation_dataset = MyDataset(tokenizer=tokenizer, **config['validation_dataset'], **config['FFT'])
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                                            **config['validation_dataloader'], shuffle=False,
                                                            collate_fn=custom_collate_fn)
    else:
        train_dataset = MyDataset(tokenizer=tokenizer, **config['train_dataset'], **config['FFT'])
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **config['train_dataloader'],
                                                       shuffle=True,
                                                       collate_fn=custom_collate_fn)

        validation_dataset = MyDataset(tokenizer=tokenizer, **config['validation_dataset'], **config['FFT'])
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                            **config['validation_dataloader'], shuffle=False,
                                                            collate_fn=custom_collate_fn)

    model = ConformerHybrid(num_vocab=tokenizer.vocab_size()+1,
                        inter_d_model=config['model']['inter_d_model'],
                        n_mel=config['model']['n_mel'],
                        time_factor=config['model']['time_factor'],
                        chunk_size=config['model']['chunk_size'],
                        context_chunk_number=config['model']['context_chunk_number'],
                        freq_dim=config['FFT']['hop_length']+1,
                        n_heads=config['model']['n_heads'],
                        n_groups=config['model']['n_groups'],
                        layer_num=config['model']['layer_num'],
                        mamba_every_n_block=config['model']['mamba_every_n_block'],
                        dropout=0.1)
    
    # ckpt = torch.load("experiments/gqa_with_mamba_FIXED_2025-11-19-14h35m/checkpoints/model_0001.tar", weights_only=False)
    # ckpt = ckpt["model"]
    # model.load_state_dict(ckpt, strict=True)
    # print("Weight loaded")

    count_parameters(model)
    model.to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['optimizer']['lr'], weight_decay=0.05)

    trainer = Trainer(config=config, model=model, tokenizer=tokenizer, optimizer=optimizer,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
                      train_sampler=train_sampler, args=args)
    # trainer._validation_epoch(1)
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
        task = Task.init(project_name='ASR', task_name="Big GQA combo with mamba") #, continue_last_task='<task_id>')
        task.upload_artifact(name='config', artifact_object=config)

    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
