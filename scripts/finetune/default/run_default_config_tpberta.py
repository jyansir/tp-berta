import numpy as np
import torch
import os
import gc
import sys
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
# import wandb
import shutil
import random
import argparse

from tqdm import trange, tqdm
from pathlib import Path

from bin import build_default_model
from lib import DataConfig, Regulator, prepare_tpberta_loaders, magnitude_regloss, calculate_metrics, make_tpberta_optimizer


def load_single_dataset(dataset_name, data_config, task_type):
    data_loader, dataset = prepare_tpberta_loaders([dataset_name], data_config, tt=task_type)
    return data_loader[0], dataset[0]

def save_result(
    args, 
    best_ev, final_test, 
    tr_losses, reg_losses, 
    ev_losses, ev_metrics, 
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
        'ev_loss': ev_losses,

    }
    if args.lamb > 0:
        saved_results['reg_loss'] = reg_losses
    with open(Path(args.result_dir) / f'{suffix}.json', 'w') as f:
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


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default='finetune_outputs')
parser.add_argument("--model_suffix", type=str, default='pytorch_models/best')
parser.add_argument("--dataset", type=str, default='HR Employee Attrition')
parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], required=True)
parser.add_argument("--lr", type=float, default=1e-5) # fixed learning rate
parser.add_argument("--weight_decay", type=float, default=0.) # no weight decay in default
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--early_stop", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lamb", type=float, default=0.) # no regularization in finetune
# parser.add_argument("--wandb", action='store_true')
args = parser.parse_args()

# keep default settings
args.result_dir = f'{args.result_dir}/{args.task}/TPBerta-default/{args.dataset}'
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

if args.task == 'binclass':
    from lib import FINETUNE_BIN_DATA as FINETUNE_DATA
    from lib import BIN_CHECKPOINT as CHECKPOINT_DIR
elif args.task == 'regression':
    from lib import FINETUNE_REG_DATA as FINETUNE_DATA
    from lib import REG_CHECKPOINT as CHECKPOINT_DIR
elif args.task == 'multiclass':
    from lib import FINETUNE_MUL_DATA as FINETUNE_DATA
    from lib import BIN_CHECKPOINT as CHECKPOINT_DIR


seed_everything(seed=42)
""" Data Preparation """
data_config = DataConfig.from_pretrained(
    CHECKPOINT_DIR, data_dir=FINETUNE_DATA,
    batch_size=64, train_ratio=0.8, 
    preproc_type='lm', pre_train=False)
(data_loader, _), dataset = load_single_dataset(args.dataset, data_config, args.task)

""" Model Preparation """
device = torch.device('cuda')
args.pretrain_dir = str(CHECKPOINT_DIR) # pre-trained TPBerta dir
model_config, model = build_default_model(args, data_config, dataset.n_classes, device, pretrain=True) # use pre-trained weights & configs
optimizer = make_tpberta_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

tot_step = 0
best_metric = -np.inf
final_test_metric = 0
no_improvement = 0
tr_task_losses, tr_reg_losses = [], []
ev_task_losses, ev_metrics = [], []
test_metrics = []
metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1
steps_per_save = 200
# wandb
# if args.wandb:
#     saved_config = {
#         'result_dir': args.result_dir,
#         'dataset': args.dataset,
#         'lr': args.lr,
#         'weight_decay': args.weight_decay,
#         'max_epochs': args.max_epochs,
#         'early_stop': args.early_stop,
#         'lamb': args.lamb,
#     }
#     wandb.init(
#         config=saved_config,
#         project='tpberta-finetune',
#         entity='zju',
#         name=f'xxx',
#         notes=f'xxx',
#         job_type='finetune'
#     )
for epoch in trange(args.max_epochs, desc='Finetuning'):
    cur_step = 0
    tr_loss = 0. # train loss
    reg_loss = 0. # regularization loss (used in pre-training but not used in finetune)
    model.train()
    for batch in tqdm(data_loader['train'], desc=f'epoch-{epoch}'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')

        optimizer.zero_grad()
        logits, _ = model(**batch)
        loss = Regulator.compute_loss(logits, labels, dataset.task_type.value)
        tr_loss += loss.cpu().item()
        if args.lamb > 0: # triplet loss used in pre-training
            reg = magnitude_regloss(labels.shape[0], data_config.num_encoder, model)
            reg_loss += reg.cpu().item()
            loss = loss + args.lamb * reg
        
        loss.backward()
        optimizer.step()
        print(f'\repoch [{epoch+1}/{args.max_epochs}] | step {cur_step+1} | avg tr loss: {tr_loss / (cur_step+1)} | avg reg loss: {reg_loss / (cur_step+1)}', end='')
        cur_step += 1
        tot_step += 1
        if tot_step % steps_per_save == 0:
            print(f'[STEP] {tot_step}: saving tmp results')
            save_result(
                args, 
                best_metric, final_test_metric, 
                tr_task_losses, tr_reg_losses, 
                ev_task_losses, ev_metrics, 
                test_metrics,
                'tmp'
            )

    tr_task_losses.append(tr_loss / cur_step)
    tr_reg_losses.append(reg_loss / cur_step)
    # if args.wandb:
    #     wandb.log({f'{args.dataset}-tr_loss': tr_task_losses[-1]}, step=tot_step)
    #     if args.lamb > 0:
    #         wandb.log({f'{args.dataset}-reg_loss': tr_reg_losses[-1]}, step=tot_step)

    # evaluating
    preds, golds, ev_loss = [], [], []
    model.eval()
    for batch in tqdm(data_loader['val'], desc='evaluate'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        with torch.no_grad():
            logits, _ = model(**batch)
            loss = Regulator.compute_loss(logits, labels, dataset.task_type.value)
        preds.append(logits.cpu())
        golds.append(labels.cpu())
        ev_loss.append(loss.cpu().item())

    ev_task_losses.append(sum(ev_loss) / len(ev_loss))
    score = calculate_metrics(
        torch.cat(golds).numpy(),
        torch.cat(preds).numpy(),
        dataset.task_type.value,
        'logits' if not dataset.is_regression else None,
        dataset.y_info
    )[metric_key] * scale
    ev_metrics.append(score)

    # testing
    preds, golds = [], []
    for batch in tqdm(data_loader['test'], desc='testing'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        with torch.no_grad():
            logits, _ = model(**batch)
        preds.append(logits.cpu())
        golds.append(labels.cpu())
    test_score = calculate_metrics(
        torch.cat(golds).numpy(),
        torch.cat(preds).numpy(),
        dataset.task_type.value,
        'logits' if not dataset.is_regression else None,
        dataset.y_info
    )[metric_key] * scale
    test_metrics.append(test_score)

    # if args.wandb:
    #     wandb.log({
    #         f'{args.dataset}-ev_loss': ev_task_losses[-1],
    #         f'{args.dataset}-ev_metric': ev_metrics[-1],
    #     }, step=tot_step)

    print()
    print(f'[Eval] {metric_key}: {score} | [Test] {metric_key}: {test_score}')
    if score > best_metric:
        best_metric = score
        final_test_metric = test_score
        no_improvement = 0
        print("best result")
    else:
        no_improvement += 1
    if args.early_stop > 0 and no_improvement == args.early_stop:
        print('early stopping')
        break

# if args.wandb:
#     wandb.finish()
save_result(
    args, 
    best_metric, final_test_metric, 
    tr_task_losses, tr_reg_losses, 
    ev_task_losses, ev_metrics, 
    test_metrics,
    'finish'
)
