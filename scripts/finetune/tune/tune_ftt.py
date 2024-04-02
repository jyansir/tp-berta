import os
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
import rtdl
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import optuna

from lib import DataConfig, data_preproc, prepare_tensors, make_optimizer, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='configs/tuned')
    parser.add_argument("--dataset", type=str, default='Iris')
    parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stop", type=int, default=16) # FT-Transformer settings
    args = parser.parse_args()

    args.output = f'{args.output}/{args.task}/ftt/{args.dataset}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args

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


"""args"""
device = torch.device('cuda')
args = get_training_args()
seed_everything(seed=42)

""" prepare Datasets and Dataloaders """
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

if dataset.X_num is None: # FT-Transformer official implementation requires at least one numerical features
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

""" Hyper-parameter Spaces """
model_param_spaces = {
    "FTTransformer": {
        "attention_dropout": (0, 0.5, 'uniform'),
        "d_ffn_factor": (2/3, 8/3, 'uniform'),
        "d_token": (64,256,8,'int'),
        "ffn_dropout": (0, 0.5, 'uniform'),
        "n_blocks": (1,3, 'int'),
        "residual_dropout": (0, 0.5, 'uniform'),
    },
}
training_param_spaces = {
    'FTTransformer': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': (1e-6, 1e-3, 'loguniform'),
        'optimizer': 'adamw'
    },
}

""" Metric Settings """
metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1

def get_model_training_params(trial):
    model_args = model_param_spaces['FTTransformer']
    training_args = {
        'batch_size': batch_size,
        'eval_batch_size': val_batch_size,
        **training_param_spaces['FTTransformer'],
    }
    model_params = {}
    training_params = {}
    for param, value in model_args.items():
        if isinstance(value, tuple):
            suggest_type = value[-1]
            if suggest_type != 'categorical':
                model_params[param] = eval(f'trial.suggest_{suggest_type}')(param, *value[:-1])
            else:
                model_params[param] = trial.suggest_categorical(param, choices=value[0])
        else:
            model_params[param] = value
    for param, value in training_args.items():
        if isinstance(value, tuple):
            suggest_type = value[-1]
            if suggest_type != 'categorical':
                training_params[param] = eval(f'trial.suggest_{suggest_type}')(param, *value[:-1])
            else:
                training_params[param] = trial.suggest_categorical(param, choices=value[0])
        else:
            training_params[param] = value
    return model_params, training_params

def objective(trial):
    cfg_model, cfg_training = get_model_training_params(trial)
    """set default"""
    cfg_model.setdefault('kv_compression_ratio', None)
    cfg_model.setdefault('kv_compression_sharing', None)
    cfg_model['ffn_d_hidden'] = int(cfg_model['d_token'] * cfg_model.pop('d_ffn_factor'))
    cfg_model = {
        'n_num_features': dataset.n_num_features,
        'd_out': d_out,
        'cat_cardinalities': dataset.get_category_sizes('train'),
        **cfg_model
    }
    model = rtdl.FTTransformer.make_baseline(**cfg_model).to(device)
    """Optimizers"""
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = make_optimizer(
        cfg_training['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        cfg_training['lr'],
        cfg_training['weight_decay'],
    )

    best_val_score = train(model, optimizer)
    return best_val_score


def train(model, optimizer):
    """Loss Function"""
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if dataset.is_binclass
        else F.cross_entropy
        if dataset.is_multiclass
        else F.mse_loss
    )

    """Utils Function"""
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

    """Training"""
    best_metric = -np.inf
    no_improvement = 0

    for epoch in range(500):
        model.train()
        for batch in tqdm(train_loader, desc=f'epoch-{epoch}'):
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )

            optimizer.zero_grad()
            loss = loss_fn(apply_model(x_num, x_cat), y)
            loss.backward()
            optimizer.step()

        scores = evaluate(['val', 'test'])
        val_score = scores['val']
        print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f}')
        if val_score > best_metric:
            best_metric = val_score
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == args.early_stop:
            print('early stop!')
            break
    return best_metric


cfg_model = model_param_spaces['FTTransformer']
const_params = {
    p: v for p, v in cfg_model.items()
    if not isinstance(v, tuple)
}
cfg_training = training_param_spaces['FTTransformer']
const_training_params = {
    p: v for p, v in cfg_training.items()
    if not isinstance(v, tuple)
}
cfg_file = f'{args.output}/cfg-tmp.json'
def save_per_iter(study, trial):
    saved_model_cfg = {**const_params}
    saved_training_cfg = {**const_training_params}
    for k in cfg_model:
        if k not in saved_model_cfg:
            saved_model_cfg[k] = study.best_trial.params.get(k)
    for k in cfg_training:
        if k not in saved_training_cfg:
            saved_training_cfg[k] = study.best_trial.params.get(k)
    saved_training_cfg = {
        'batch_size': batch_size,
        'eval_batch_size': val_batch_size,
        **saved_training_cfg
    }
    hyperparams = {
        'metric': metric_key,
        'eval_score': study.best_trial.value,
        'n_trial': study.best_trial.number,
        'dataset': args.dataset,
        'model': saved_model_cfg,
        'training': saved_training_cfg,
    }
    with open(cfg_file, 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

iterations = 100
study = optuna.create_study(direction="maximize")
study.optimize(func=objective, n_trials=iterations, callbacks=[save_per_iter])


cfg_file = f'{args.output}/cfg.json'
for k in cfg_model:
    if k not in const_params:
        const_params[k] = study.best_params.get(k)
for k in cfg_training:
    if k not in const_training_params:
        const_training_params[k] = study.best_params.get(k)
const_training_params = {
    'batch_size': batch_size,
    'eval_batch_size': val_batch_size,
    **const_training_params
}

hyperparams = {
    'metric': metric_key,
    'eval_score': study.best_value,
    'n_trial': study.best_trial.number,
    'dataset': args.dataset,
    'model': const_params,
    'training': const_training_params,
}
with open(cfg_file, 'w') as f:
    json.dump(hyperparams, f, indent=4, ensure_ascii=False)