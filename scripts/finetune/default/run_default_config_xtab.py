import os
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
import torch
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import warnings
warnings.filterwarnings('ignore')

from lib import DataConfig, data_preproc, calculate_metrics
from lib import BIN_CHECKPOINT as CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_epochs', type=int, default=3)
parser.add_argument('--num_pretrain_rounds', type=str, choices=['0', '1k', '2k'], required=True)
parser.add_argument('--pretrained_ckpts', default='./checkpoints/xtab-checkpoint')
parser.add_argument("--result_dir", type=str, default='finetune_outputs')
parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
parser.add_argument("--dataset", type=str, default='train_1811_Pokemon-with-stats-Generation-8')
args = parser.parse_args()

args.pretrained_ckpts = str(Path(args.pretrained_ckpts) / f"iter_{args.num_pretrain_rounds}.ckpt")
args.result_dir = str(Path(args.result_dir) / args.task / f'XTab-default-{args.num_pretrain_rounds}')

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

seed_everything(args.seed)

args.result_dir = os.path.join(args.result_dir, args.dataset)
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
    preproc_type=args.model, pre_train=False)
dataset = data_preproc(args.dataset, no_str=True, config=data_config, tt=args.task)


df_train = {k: [] for k in dataset.feature_names['num']}
df_val = {k: [] for k in dataset.feature_names['num']}
df_test = {k: [] for k in dataset.feature_names['num']}

for i, k in enumerate(dataset.feature_names['num']):
    df_train[k] = dataset.X_num['train'][:, i]

for i, k in enumerate(dataset.feature_names['num']):
    df_val[k] = dataset.X_num['val'][:, i]

for i, k in enumerate(dataset.feature_names['num']):
    df_test[k] = dataset.X_num['test'][:, i]

df_train = pd.DataFrame(df_train)
y_train = pd.Series(dataset.y['train'])
y_val = pd.Series(dataset.y['val'])
y_test = pd.Series(dataset.y['test'])



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

label = "label"
df_train[label] = y_train
df_val[label] = y_val
df_test[label] = y_test

hyperparameters = {} 
hyperparameters['FT_TRANSFORMER'] = {
    "env.per_gpu_batch_size": args.batch_size,
    "env.num_workers": 0,
    "env.num_workers_evaluation": 0,
    "optimization.max_epochs": args.max_epochs,
    'finetune_on': args.pretrained_ckpts,
}
print(hyperparameters)

metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]

predictor = TabularPredictor(label=label,
                                # eval_metric=metric_key,
                                # eval_metric="root_mean_squared_error",
                                )

predictor.fit(
    train_data=df_train,
    hyperparameters=hyperparameters,
    time_limit=60,
    keep_only_best = True,
    fit_weighted_ensemble = False,
)

probabilities = predictor.predict_proba(df_test, as_pandas=False)
probabilities_val = predictor.predict_proba(df_val, as_pandas=False)

scale = 1 if not dataset.is_regression else -1
test_score = calculate_metrics(
    y_test.values,
    (
        probabilities if args.task == 'multiclass' 
        else probabilities[:, 1] 
        if args.task == 'binclass' 
        else probabilities
    ),
    dataset.task_type.value,
    'probs' if not dataset.is_regression else None,
    dataset.y_info
)[metric_key] * scale
val_score = calculate_metrics(
    y_val.values,
    (
        probabilities_val if args.task == 'multiclass' 
        else probabilities_val[:, 1] 
        if args.task == 'binclass' 
        else probabilities_val
    ),
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

shutil.rmtree(predictor.path)
print('rm: ', predictor.path)
