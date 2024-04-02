import os
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
import numpy as np
import torch
import argparse
import pandas as pd


from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from lib import DataConfig, data_preproc, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default='finetune_outputs')
parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
parser.add_argument("--dataset", type=str, default='train_1811_Pokemon-with-stats-Generation-8')
args = parser.parse_args()

if args.task == 'binclass':
    from lib import FINETUNE_BIN_DATA as FINETUNE_DATA
elif args.task == 'regression':
    from lib import FINETUNE_REG_DATA as FINETUNE_DATA
elif args.task == 'multiclass':
    from lib import FINETUNE_MUL_DATA as FINETUNE_DATA

args.result_dir = f'{args.result_dir}/{args.task}/tabnet-default/{args.dataset}'
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

data_config = DataConfig.from_pretrained(
    CHECKPOINT_DIR, data_dir=FINETUNE_DATA,
    batch_size=64, train_ratio=0.8, 
    preproc_type='ftt', pre_train=False)
dataset = data_preproc(args.dataset, no_str=True, config=data_config, tt=args.task)

num_cols, cat_cols = dataset.feature_names['num'], dataset.feature_names['cat']

df_train, df_val, df_test = [], [], []
if len(cat_cols) > 0:
    df_train.append(dataset.X_cat['train'])
    df_val.append(dataset.X_cat['val'])
    df_test.append(dataset.X_cat['test'])
if len(num_cols) > 0:
    df_train.append(dataset.X_num['train'])
    df_val.append(dataset.X_num['val'])
    df_test.append(dataset.X_num['test'])
cat_idxs = [] if dataset.X_cat is None else list(range(dataset.n_cat_features))
cat_dims = [] if dataset.X_cat is None else dataset.get_category_sizes('train')

df_train = np.concatenate(df_train, axis=1)
df_val = np.concatenate(df_val, axis=1)
df_test = np.concatenate(df_test, axis=1)

y_train = dataset.y['train']
y_val = dataset.y['val']
y_test = dataset.y['test']

if args.task == 'regression':
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

if args.task == 'binclass':
    clf = TabNetClassifier() 
    clf.fit(
        df_train, y_train,
        eval_set=[(df_val, y_val)]
    )
elif args.task == 'multiclass':
    clf = TabNetClassifier(output_dim=dataset.n_classes) 
    clf.fit(
        df_train, y_train,
        eval_set=[(df_val, y_val)]
    )
else:
    clf = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims)
    clf.fit(
        df_train, y_train,
        eval_set=[(df_val, y_val)]
    )

if args.task != 'multiclass':
    ypred = clf.predict(df_test)
    ypred_val = clf.predict(df_val)
else:
    ypred = clf.predict_proba(df_test)
    ypred_val = clf.predict_proba(df_val)

metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1
val_score = calculate_metrics(
    y_val,
    ypred_val,
    dataset.task_type.value,
    'probs' if not dataset.is_regression else None,
    dataset.y_info
)[metric_key] * scale
test_score = calculate_metrics(
    y_test,
    ypred,
    dataset.task_type.value,
    'probs' if not dataset.is_regression else None,
    dataset.y_info
)[metric_key] * scale

def save_result(
    args, 
    best_ev,
    final_test, 
    suffix
):
    saved_results = {
        'args': vars(args),
        'device': torch.cuda.get_device_name(),
        'best_eval_score': best_ev,
        'final_test_score': final_test,

    }
    with open(Path(args.result_dir) / f'{suffix}.json', 'w') as f:
        json.dump(saved_results, f, indent=4)

save_result(args, val_score, test_score, 'finish')