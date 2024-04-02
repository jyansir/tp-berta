import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import trange, tqdm

import torch
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib

from bin import build_default_model
from lib import DataConfig, MTLoader, Regulator, prepare_tpberta_loaders, magnitude_regloss


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default='./pretrain_outputs', help='Path to pre-trained models')

# model config
parser.add_argument("--base_model_dir", type=str, default='./checkpoints/roberta-base', help='Path to base model weights & configs')
parser.add_argument("--max_position_embeddings", type=int, default=64)
parser.add_argument("--type_vocab_size", type=int, default=5) # keep default

# data config
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--max_feature_length", type=int, default=8)
parser.add_argument("--max_numerical_token", type=int, default=256)
parser.add_argument("--max_categorical_token", type=int, default=16)
parser.add_argument("--feature_map", type=str, default='feature_names.json', help='File for standard feature name')
parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'joint'], help='Task type of pre-trained datasets', required=True)

# training config
parser.add_argument("--batch_size", type=int, default=512, help='Batch size of pre-training, default: 512')
parser.add_argument("--early_stop", type=int, default=10000, help='Early stop based on the evaluation policy, default: no early stop')
parser.add_argument("--warmup", type=int, default=6000, help='Warm up steps')
parser.add_argument("--max_epochs", type=int, default=30, help='Pre-training max epochs, default: 30')
parser.add_argument("--peak_lr", type=float, default=3e-5, help='Maximum learning rate')
parser.add_argument("--min_lr", type=float, default=1e-6, help='Minimum learning rate')
parser.add_argument("--lamb", type=float, default=0.1, help='Weight for triplet loss regularization')
parser.add_argument("--wandb", action='store_true', help='Upload to wandb for inspection or not')

args = parser.parse_args()
# Path to pre-training results
run_id = f'TPBerta_{args.task}-BS{args.batch_size}-MaxEpoch{args.max_epochs}' # Run ID
args.result_dir = f'{args.result_dir}/{run_id}'
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)


def fetch_dataset_list(data_dir: Path):
    # read a data directory with all csv files for pre-training
    ds = [d[:-4] for d in os.listdir(data_dir) if d.endswith('.csv')]
    # filter some dirty datasets if needed
    if os.path.exists(data_dir / 'skip_datasets.json'):
        with open(data_dir / 'skip_datasets.json', 'r') as f:
            skip_datasets = json.load(f)
        ds = [d for d in ds if not any(sd in d for sd in skip_datasets)]
    return ds


""" Data Preparation """
if args.task != 'joint':
    if args.task == 'binclass':
        # Path to binary classification datasets (for pre-training)
        from lib import PRETRAIN_BIN_DATA as DATA
    elif args.task == 'regression':
        # Path to binary regression datasets (for pre-training)
        from lib import PRETRAIN_REG_DATA as DATA

    # data preprocessing config (default: 95% training, 5% evaluation)
    data_config = DataConfig.from_default(args, data_dir=DATA, train_ratio=0.95, preproc_type='lm', pre_train=True)
    # data_config.save_pretrained(args.result_dir)

    dataset_names = fetch_dataset_list(DATA) # pre-training datasets

    # prepare dataloader for each dataset
    dataloaders, datasets = prepare_tpberta_loaders(dataset_names, data_config, tt=args.task)

else: # pre-training on both binclass & regression datasets
    from lib import PRETRAIN_BIN_DATA, PRETRAIN_REG_DATA
    # the only difference between data configs of binclass & regression is the data path
    data_config_bin = DataConfig.from_default(args, data_dir=PRETRAIN_BIN_DATA, train_ratio=0.95, preproc_type='lm', pre_train=True)
    data_config_reg = DataConfig.from_default(args, data_dir=PRETRAIN_REG_DATA, train_ratio=0.95, preproc_type='lm', pre_train=True)

    dataset_names_bin = fetch_dataset_list(PRETRAIN_BIN_DATA)
    dataset_names_reg = fetch_dataset_list(PRETRAIN_REG_DATA)

    print('loading the pre-training binclass datasets')
    dataloaders_bin, datasets_bin = prepare_tpberta_loaders(dataset_names_bin, data_config_bin, tt='binclass')
    print('loading the pre-training regression datasets')
    dataloaders_reg, datasets_reg = prepare_tpberta_loaders(dataset_names_reg, data_config_reg, tt='regression')
    
    # merge list of dataloaders and datasets
    dataloaders = dataloaders_bin + dataloaders_reg
    datasets = datasets_bin + datasets_reg
    data_config = data_config_bin # same as data_config_reg except data path, do not affect the subsequent training


train, val, task_types = [], [], []
for i, (data_loader, task_type) in enumerate(dataloaders):
    train.append(data_loader['train']) # train loaders
    val.append(data_loader['val']) # val loaders
    task_types.append(datasets[i].task_type.value)


""" Model Preparation """
# model
device = torch.device('cuda')
num_classes = [d.n_classes for d in datasets] # class numbers for task-specific heads
model_config, model = build_default_model(args, data_config, num_classes, device)
# model_config.save_pretrained(args.result_dir) # for downstream load

# optimizer and scheduler
regulator = Regulator.from_default(
    datasets, args, 
    steps_per_epoch=sum(len(x[0]['train']) for x in dataloaders), # training steps per epoch
    eval_times_per_epoch=5 # eval times per epoch
)
regulator.set_optimizer_and_scheduler(model)


""" Training Loops """
tot_step = 0
regulator.trainer_start(args, model)
data_config.save_pretrained(args.result_dir) # save data preprocessing config
model_config.save_pretrained(args.result_dir) # save model config
for epoch in trange(regulator.max_epochs, desc='pre-training process'):
    train_loader = MTLoader(train) # a wrapped multi-task dataloader for multiple train loaders
    regulator.epoch_start(epoch, model)
    for dataset_idx, batch in tqdm(train_loader, desc='training'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        # train
        regulator.step_start(epoch, tot_step, model)
        logits, _ = model(dataset_idx=dataset_idx, **batch)

        d = regulator.datasets[dataset_idx]
        bs = labels.shape[0]
        # DEBUG: if output side trigger error, the index is likely to be out of boundary
        # CheckList: 1. label index in loss func or 2. index for vocab / position embeddings
        task_loss = regulator.compute_loss(logits, labels, d.task_type.value)
        # print(task_loss) # DEBUG: for error inspection when task loss is abnormal

        # Regularization: magnitude-aware triplet loss
        reg_loss = magnitude_regloss(bs, data_config.num_encoder, model)
        loss = task_loss + args.lamb * reg_loss
        loss.backward()

        # update loss
        regulator.update_tr_loss(dataset_idx, bs, task_loss.cpu().item(), reg_loss.cpu().item())
        regulator.step_end(epoch, tot_step, model, val) # automatically evaluate & save model (default: based on eval loss)
        tot_step += 1
    regulator.epoch_end(epoch, model, val)
regulator.trainer_end(args, model) # copy weights of best & last models

# ASSIGN the result path as CHECKPOINT path in ./lib/env.py
# or copy the result path as follows
# if args.task == 'binclass':
#     from lib import BIN_CHECKPOINT as CHECKPOINT
# elif args.task == 'regression':
#     from lib import REG_CHECKPOINT as CHECKPOINT
# else:
#     from lib import JOINT_CHECKPOINT as CHECKPOINT

# if os.path.exists(CHECKPOINT):
#     print('Old checkpoint exists: ', str(CHECKPOINT))
#     shutil.rmtree(CHECKPOINT)
#     print('removed')
# print('copying the resulting model to the CHECKPOINT path')
# try:
#     shutil.copytree(args.result_dir, CHECKPOINT)
#     print('success!')
#     delete_copy = input("Want to delete the pre-training result directory? (y/n)")
#     if delete_copy.lower() == 'y':
#         shutil.rmtree(args.result_dir)
#         print('clear the result result directory, you can find them in: ', str(CHECKPOINT))
# except:
#     print('copying failed, please manually assign the result directory as CHECKPOINT path in lib/env.py')
