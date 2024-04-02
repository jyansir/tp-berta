import os
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import optuna

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from lib import DataConfig, data_preproc, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['xgboost', 'catboost', 'tabnet'])
    parser.add_argument("--output", type=str, default='configs/tuned')
    parser.add_argument("--dataset", type=str, default='train_1811_Pokemon-with-stats-Generation-8')
    parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
    args = parser.parse_args()

    args.output = f'{args.output}/{args.task}/{args.model}/{args.dataset}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args

def save_result(
    args, 
    model_cfgs,
    best_ev, final_test, 
    suffix
):
    saved_results = {
        'args': vars(args),
        'device': torch.cuda.get_device_name(),
        'configs': model_cfgs,
        'best_eval_score': best_ev,
        'final_test_score': final_test,
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

preproc_type = 'ftt' if args.model == 'tabnet' else args.model
data_config = DataConfig.from_pretrained(
    CHECKPOINT_DIR, data_dir=FINETUNE_DATA,
    batch_size=64, train_ratio=0.8, 
    preproc_type=preproc_type, pre_train=False)
dataset = data_preproc(args.dataset, data_config, no_str=True, tt=args.task)

prediction_type = None if dataset.is_regression else 'probs'
metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1


if dataset.X_cat is None:
    Xs = {k: dataset.X_num[k] for k in ['train', 'val', 'test']}
else:
    if args.model == 'tabnet':
        Xs = {k: np.concatenate((dataset.X_cat[k], dataset.X_num[k]), axis=1) for k in ['train', 'val', 'test']}
    else:
        Xs = {k: np.concatenate((dataset.X_num[k], dataset.X_cat[k]), axis=1) for k in ['train', 'val', 'test']}
ys = {k: dataset.y[k] for k in ['train', 'val', 'test']}

if args.model == 'tabnet':
    cat_idxs = [] if dataset.X_cat is None else list(range(dataset.n_cat_features))
    cat_dims = [] if dataset.X_cat is None else dataset.get_category_sizes('train')
    if args.task_type == 'regression':
        ys = {k: ys[k].reshape(-1, 1) for k in ['train', 'val', 'test']}
  

""" Hyper-parameter Spaces """
model_param_spaces = {
    'xgboost': {
        "alpha": (1e-8, 1e2, 'loguniform'),
        "booster": "gbtree",
        "colsample_bylevel": (0.5, 1, 'uniform'),
        "colsample_bytree": (0.5, 1, 'uniform'),
        "gamma": (1e-8, 1e2, 'loguniform'),
        "lambda": (1e-8, 1e2, 'loguniform'),
        "learning_rate": (1e-5, 1, 'loguniform'),
        "max_depth": (3, 10, 'int'),
        "min_child_weight": (1e-8, 1e5, 'loguniform'),
        "n_estimators": 2000,
        "n_jobs": -1,
        "subsample": (0.5, 1, 'uniform'),
        "tree_method": "gpu_hist"
    },
    'catboost': {
        # base config
        'iterations': 2000,
        'metric_period': 10,
        'od_pval': 0.001,
        'task_type': 'GPU',
        'devices': '0',
        'thread_count': 1,
        'random_seed': 42,
        'gpu_ram_part': 0.8, # GPU utilization
        # search space
        "l2_leaf_reg": (1.0, 10.0, 'loguniform'),
        "bagging_temperature": (0.0, 1.0, 'uniform'),
        "depth": (3, 10, 'int'),
        "leaf_estimation_iterations": (1, 10, 'int'),
        "learning_rate": (1e-5, 1, 'loguniform'),
    },
    'tabnet': {
        "n_steps": (3, 10, 'int'),
        "n_a": ([8, 16, 32, 64, 128], 'categorical'),
        "n_d": ([8, 16, 32, 64, 128], 'categorical'),
        "gamma": (1.0, 2.0, 'uniform'),
        "lambda_sparse": (1e-6, 1e-1, 'loguniform'),
        "lr": (1e-3, 1e-2, 'loguniform'),
    }
}
training_param_spaces = {
    'xgboost': {
        "early_stopping_rounds": 20,
        "verbose": False
    },
    'catboost': {
        "early_stopping_rounds": 50,
        'logging_level': 'Verbose'
    },
    'tabnet': {
        "batch_size": 256,
    }
}

def get_model_training_params(trial):
    model_args = model_param_spaces[args.model]
    training_args = training_param_spaces[args.model]
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

""" Tree Model """
def objective(trial):
    cfg_model, cfg_training = get_model_training_params(trial)

    if args.model == 'xgboost':
        if dataset.is_regression:
            model = XGBRegressor(**cfg_model, random_state=42)
            predict = model.predict
        else:
            model = XGBClassifier(**cfg_model, random_state=42, disable_default_eval_metric=True)
            if dataset.is_multiclass:
                predict = model.predict_proba
                cfg_training['eval_metric'] = 'merror'
            else:
                predict = lambda x: model.predict_proba(x)[:, 1]
                cfg_training['eval_metric'] = 'error'
        
        model.fit(
            Xs['train'],
            ys['train'],
            eval_set=[(Xs['val'], ys['val'])],
            **cfg_training,
        )
    elif args.model == 'catboost':
        if dataset.is_regression:
            model = CatBoostRegressor(**cfg_model)
            predict = model.predict
        else:
            model = CatBoostClassifier(**cfg_model, eval_metric='Accuracy')
            predict = (
                model.predict_proba
                if dataset.is_multiclass
                else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
            )
        model.fit(
            Xs['train'],
            ys['train'],
            **cfg_training,
            eval_set=(Xs['val'], ys['val']),
        )
    elif args.model == 'tabnet':
        cfg_model['cat_idxs'] = cat_idxs
        cfg_model['cat_dims'] = cat_dims
        cfg_model[ "optimizer_params"] = {
            "lr": cfg_model.pop("lr"),
        }
        if dataset.is_regression:
            model = TabNetRegressor(**cfg_model)
            predict = model.predict
        else:
            model = TabNetClassifier(**cfg_model)
            predict = (
                model.predict_proba
                if dataset.is_multiclass
                else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
            )
        model.fit(
            Xs['train'],
            ys['train'],
            eval_set=[(Xs['val'], ys['val'])],
            **cfg_training,
        )
    else:
        raise NotImplementedError
    preds = predict(Xs['val'])


    val_score = calculate_metrics(
        ys['val'],
        preds,
        dataset.task_type.value,
        'probs' if not dataset.is_regression else None,
        dataset.y_info
    )[metric_key] * scale
    return val_score
        

cfg_model = model_param_spaces[args.model]
const_params = {
    p: v for p, v in cfg_model.items()
    if not isinstance(v, tuple)
}
cfg_training = training_param_spaces[args.model]
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