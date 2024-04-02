import os
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib import DataConfig, data_preproc, prepare_tensors, make_optimizer, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR
from bin import MLP, AutoInt, DCNv2, SAINT

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['mlp', 'autoint', 'dcnv2', 'saint'])
    parser.add_argument("--output", type=str, default='finetune_outputs')
    parser.add_argument("--dataset", type=str, default='train_1748_Sales_DataSet_of_SuperMarket')
    parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stop", type=int, default=16) # FT-Transformer settings
    args = parser.parse_args()

    args.output = f'{args.output}/{args.task}/{args.model}-tuned/{args.dataset}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    cfg_file = f'configs/tuned/{args.task}/{args.model}/{args.dataset}/cfg.json'
    if not os.path.exists(cfg_file):
        shutil.rmtree(args.output)
        raise AssertionError(f'{args.model}-{args.dataset} tuned config missing')
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)
    
    return args, cfg

def save_result(
    args, 
    best_ev, final_test, 
    tr_losses,
    ev_metrics, 
    test_metrics, 
    suffix
):
    saved_results = {
        'args': vars(args),
        'device': torch.cuda.get_device_name(),
        'best_eval_score': best_ev,
        'final_test_score': final_test,
        'ev_metric': ev_metrics,
        'test_metric': test_metrics,
        'tr_loss': tr_losses,
    }
    with open(Path(args.output) / f'{suffix}.json', 'w') as f:
        json.dump(saved_results, f, indent=4)

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# args
device = torch.device('cuda')
args, cfg = get_training_args()
seed_everything(seed=42)

# prepare Datasets and Dataloaders
if args.task == 'binclass':
    from lib import FINETUNE_BIN_DATA as FINETUNE_DATA
elif args.task == 'regression':
    from lib import FINETUNE_REG_DATA as FINETUNE_DATA
elif args.task == 'multiclass':
    from lib import FINETUNE_MUL_DATA as FINETUNE_DATA

data_config = DataConfig.from_pretrained(
    CHECKPOINT_DIR, data_dir=FINETUNE_DATA,
    batch_size=64, train_ratio=0.8, 
    preproc_type='ftt', pre_train=False)
dataset = data_preproc(args.dataset, data_config, no_str=True, tt=args.task)

if args.model == 'saint' and dataset.X_num is None: # SAINT original implementation requires at least one numerical features
    new_Xnum = {k: v[:, :1].astype(np.float32) for k, v in dataset.X_cat.items()} # treat the first categorical one as numerical
    new_Xcat = {k: v[:, 1:] for k, v in dataset.X_cat.items()}
    from dataclasses import replace
    dataset = replace(dataset, X_num=new_Xnum, X_cat=new_Xcat)

d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

batch_size = args.batch_size
val_batch_size = 128


# data loaders
check_list = {}
for x in ['X_num', 'X_cat']:    
    check_list[x] = False if eval(x) is None else True

data_list = [x for x in [X_num, X_cat, ys] if x is not None]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

# Hyper-parameter Spaces
model_args = cfg['model']
training_args = cfg['training']

# Metric Settings
metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1


def process_mlp_params(params):
    d_layers = []
    for i in range(params['n_layers']):
        if i == 0:
            d_layers.append(params['first_dim'])
        elif i == params['n_layers'] - 1 and params['n_layers'] > 1:
            d_layers.append(params['last_dim'])
        else:
            d_layers.append(params['mid_dim'])
    params['d_layers'] = d_layers
    del params['n_layers'], params['first_dim'], params['mid_dim'], params['last_dim']
    return params


cats = dataset.get_category_sizes('train')
if len(cats) == 0:
    cats = None

# set default
if args.model == 'mlp':
    model_args.setdefault('d_embedding', None)
    model_args = process_mlp_params(model_args)
    model = MLP(
        d_in=dataset.n_num_features, 
        categories=cats,
        d_out=d_out,
        **model_args
    ).to(device)
elif args.model == 'autoint':
    model_args.setdefault('kv_compression', None)
    model_args.setdefault('kv_compression_sharing', None)
    model = AutoInt(
        d_numerical=dataset.n_num_features,
        categories=cats,
        d_out=d_out,
        **model_args
    ).to(device)
elif args.model == 'dcnv2':
    model = DCNv2(
        d_in=dataset.n_num_features, 
        categories=cats,
        d_out=d_out,
        **model_args
    ).to(device)
elif args.model == 'saint':
    model = SAINT(
        d_numerical=dataset.n_num_features,
        categories=cats,
        d_out=d_out,
    ).to(device)

# Optimizers
if args.model in ['autoint', 'ftt']:
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = make_optimizer(
        training_args['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        training_args['lr'],
        training_args['weight_decay'],
    )
else:
    optimizer = make_optimizer(
        training_args['optimizer'],
        model.parameters(),
        training_args['lr'],
        training_args['weight_decay'],
    )


def train_predict(model, optimizer):
    # Loss Function
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if dataset.is_binclass
        else F.cross_entropy
        if dataset.is_multiclass
        else F.mse_loss
    )

    # Utils Function
    def apply_model(x_num, x_cat):
        logits = model(x_num, x_cat)
        if logits.ndim == 2:
            return logits.squeeze(-1)
        return logits

    @torch.inference_mode()
    def evaluate(parts):
        model.eval()
        results = {}
        for part in parts:
            assert part in ['train', 'val', 'test']
            golds, preds = [], []
            for batch in dataloaders[part]:
                if check_list['X_num']:
                    x_num = batch[0]
                    x_cat = None
                    if check_list['X_cat']:
                        x_cat = batch[1]
                else:
                    x_num = None
                    x_cat = batch[0]
                y = batch[-1]
                preds.append(apply_model(x_num, x_cat).cpu())
                golds.append(y.cpu())
            score = calculate_metrics(
                torch.cat(golds).numpy(),
                torch.cat(preds).numpy(),
                dataset.task_type.value,
                'logits' if not dataset.is_regression else None,
                dataset.y_info
            )[metric_key] * scale
            results[part] = score
        return results

    # Training
    tot_step = 0
    best_metric = -np.inf
    final_test_metric = 0
    no_improvement = 0
    tr_task_losses = []
    ev_metrics = []
    test_metrics = []

    for epoch in range(500):
        tot_tr_loss = 0
        model.train()
        for iteration, batch in enumerate(train_loader):
            if check_list['X_num']:
                x_num = batch[0]
                x_cat = None
                if check_list['X_cat']:
                    x_cat = batch[1]
            else:
                x_num = None
                x_cat = batch[0]
            y = batch[-1]

            optimizer.zero_grad()
            loss = loss_fn(apply_model(x_num, x_cat), y)
            loss.backward()
            optimizer.step()
            tot_step += 1
            tot_tr_loss += loss.cpu().item()

        tr_task_losses.append(tot_tr_loss / (iteration + 1))
        scores = evaluate(['val', 'test'])
        val_score, test_score = scores['val'], scores['test']
        ev_metrics.append(val_score)
        test_metrics.append(test_score)
        print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f}')
        if val_score > best_metric:
            best_metric = val_score
            final_test_metric = test_score
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == args.early_stop:
            print('early stop!')
            break
    return tr_task_losses, ev_metrics, test_metrics, best_metric, final_test_metric


tr_task_losses, ev_metrics, test_metrics, best_metric, final_test_metric = \
    train_predict(model, optimizer)

save_result(
    args,
    best_ev=best_metric, final_test=final_test_metric,
    tr_losses=tr_task_losses, ev_metrics=ev_metrics, test_metrics=test_metrics,
    suffix='finish'
)
