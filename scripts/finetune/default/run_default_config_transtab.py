import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
import torch
import argparse
import transtab
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from lib import DataConfig, data_preproc, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default='finetune_outputs')
parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
parser.add_argument("--dataset", type=str, default='Iris')
args = parser.parse_args()

args.result_dir = f'{args.result_dir}/{args.task}/transtab-default/{args.dataset}'
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

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
dataset = data_preproc(args.dataset, no_str=True, config=data_config, tt=args.task)

num_cols, cat_cols = dataset.feature_names['num'], dataset.feature_names['cat']

df_train = {k: [] for k in num_cols + cat_cols}
df_val = {k: [] for k in num_cols + cat_cols}
df_test = {k: [] for k in num_cols + cat_cols}

for i, k in enumerate(num_cols):
    df_train[k] = dataset.X_num['train'][:, i]

for i, k in enumerate(num_cols):
    df_val[k] = dataset.X_num['val'][:, i]

for i, k in enumerate(num_cols):
    df_test[k] = dataset.X_num['test'][:, i]

if len(cat_cols) > 0:
    for i, k in enumerate(cat_cols):
        df_train[k] = dataset.X_cat['train'][:, i]

    for i, k in enumerate(cat_cols):
        df_val[k] = dataset.X_cat['val'][:, i]

    for i, k in enumerate(cat_cols):
        df_test[k] = dataset.X_cat['test'][:, i]    

df_train = pd.DataFrame(df_train)
df_val = pd.DataFrame(df_val)
df_test = pd.DataFrame(df_test)

y_train = pd.Series(dataset.y['train'])
y_val = pd.Series(dataset.y['val'])
y_test = pd.Series(dataset.y['test'])

trainset = [(df_train, y_train)]
valset = [(df_val, y_val)]
testset = [(df_test, y_test)]
allset = [(df_train, y_train), (df_val, y_val), (df_test, y_test)]

bin_cols = []
for col in cat_cols:
    if len(df_train[col].unique()) == 2:
        bin_cols.append(col)
cat_cols = list(set(cat_cols) - set(bin_cols))

# build transtab classifier model
if args.task == 'multiclass':
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols, num_class=dataset.n_classes)
else:
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

# specify training arguments, take validation loss for early stopping
training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint'
}
    
# start training, take the validation loss on average for evaluation
transtab.train(model, trainset, valset, **training_arguments)

# make predictions on the first dataset 'credit-g'
x_test, y_test = testset[0]
# ypred = transtab.predict(model, testset[0][0], testset[0][1])
ypred = transtab.predict(model, df_test, y_test)
ypred_val = transtab.predict(model, df_val, y_val)


metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1
val_score = calculate_metrics(
    y_val.values,
    ypred_val,
    dataset.task_type.value,
    'probs' if not dataset.is_regression else None,
    dataset.y_info
)[metric_key] * scale
test_score = calculate_metrics(
    y_test.values,
    ypred,
    dataset.task_type.value,
    'probs' if not dataset.is_regression else None,
    dataset.y_info
)[metric_key] * scale

# print(test_score)


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