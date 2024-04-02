import os
import json
import time
import wandb
import shutil
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import transformers
from transformers.trainer_pt_utils import get_parameter_names

from .data import Dataset2
from .metrics import calculate_metrics
from .data_utils import MTLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1e-6)


def make_tpberta_optimizer(model: Union[nn.Module, nn.DataParallel], lr, weight_decay, beta1=0.9, beta2=0.98, eps=1e-6):
    model = model.module if isinstance(model, nn.DataParallel) else model
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(beta1, beta2), eps=eps)
    return optimizer


@dataclass
class Regulator:
    datasets: List[Dataset2]
    # metric recorder
    train_losses: Optional[Dict[str, List[float]]] = None # data-specific
    eval_losses: Optional[Dict[str, List[float]]] = None # data-specific
    avg_train_losses: List[float] = field(default_factory=list)
    avg_eval_losses: List[float] = field(default_factory=list)
    reg_losses: List[float] = field(default_factory=list)
    mlm_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)

    # train_metrics: Optional[Dict[str, List[float]]] = None
    eval_metrics: Optional[Dict[str, List[float]]] = None

    loss_holders: Optional[Dict[str, AverageMeter]] = None # train loss holder; eval loss holder is built once evaluate
    reg_loss_holder: AverageMeter = field(default_factory=AverageMeter) # reg loss holder
    mlm_loss_holder: AverageMeter = field(default_factory=AverageMeter) # mlm loss holder
    best_metric: float = np.inf
    best_epoch: int = -1
    best_step: int = None
    run_time_holder: Optional[AverageMeter] = field(default_factory=AverageMeter)
    start_time: float = None
    # optimizer and scheduler args
    peak_lr: float = 6e-4
    min_lr: float = 1e-6
    warmup: int = 1000 # warmup steps
    warmup_policy: str = 'step'
    max_epochs: int = 200 #
    steps_per_epoch: int = None
    lr_decay: str = 'linear'
    weight_decay: float = 0.01
    eps: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.98
    # optimizer holder
    optimizer: Optional[AdamW] = None
    scheduler: Optional[LambdaLR] = None
    # training args
    output_dir: str = None
    overwrite: bool = True
    early_stop: int = 32
    use_early_stop: bool = False
    eval_policy: str = 'step' # evaluation policy (step / epoch)
    metric_policy: str = 'loss'
    eval_steps: int = 200 # used if eval_policy is `step`
    save_policy: str = 'step'
    print_policy: str = 'step'
    max_best_save: int = 5 # max save number of best models (i.e., retain the recent 5 best checkpoints)
    max_save: int = 3 # max save number of recent models (i.e., save the recent 3 checkpoints)
    use_wandb: bool = False # upload wandb
    
    @classmethod
    def from_default(cls, dataset_names: List[Dataset2], args=None, **kwargs):
        assert 'steps_per_epoch' in kwargs
        print('train steps per epoch: ', kwargs['steps_per_epoch'])
        if 'eval_times_per_epoch' in kwargs:
            kwargs['eval_steps'] = kwargs['steps_per_epoch'] // kwargs.pop('eval_times_per_epoch')
        used_keys = ['peak_lr', 'min_lr', 'warmup', 'max_epochs', 'early_stop']
        update_kws = {} if args is None else {k: vars(args)[k] for k in used_keys}
        return Regulator(
            datasets=dataset_names,
            train_losses={d.name: [] for d in dataset_names},
            eval_losses={d.name: [] for d in dataset_names},
            # train_metrics={d.name: [] for d in dataset_names},
            eval_metrics={d.name: [] for d in dataset_names},
            loss_holders={d.name: AverageMeter() for d in dataset_names},
            # reg_loss_holder=AverageMeter(),
            # run_time_holder=AverageMeter(),
            output_dir=args.result_dir if args is not None else './results',
            use_wandb=args.wandb,
            **update_kws,
            **kwargs,
        )
    
    def set_optimizer_and_scheduler(self, model: Union[nn.Module, nn.DataParallel]):
        model = model.module if isinstance(model, nn.DataParallel) else model
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.peak_lr, betas=(self.beta1, self.beta2), eps=self.eps)
        if self.lr_decay == 'linear':
            if self.warmup_policy == 'step':
                max_steps = self.steps_per_epoch * self.max_epochs
                _lambda = (
                    lambda step: (self.min_lr + (self.peak_lr - self.min_lr) * step / self.warmup) / self.peak_lr if step < self.warmup
                    else (self.peak_lr - (self.peak_lr - self.min_lr) * (step - self.warmup) / (max_steps - self.warmup)) / self.peak_lr
                )
            else:
                _lambda = (
                    lambda epoch: (self.min_lr + (self.peak_lr - self.min_lr) * epoch / self.warmup) / self.peak_lr if epoch < self.warmup
                    else (self.peak_lr - (self.peak_lr - self.min_lr) * (epoch - self.warmup) / (self.max_epochs - self.warmup)) / self.peak_lr
                )
            scheduler = LambdaLR(optimizer, _lambda)
        else:
            raise NotImplementedError
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        return optimizer, scheduler
    
    @property
    def n_datasets(self):
        return len(self.datasets)
    
    @property
    def avg_loss(self):
        # avg train loss
        losses = [lh.avg for lh in self.loss_holders.values() if lh.avg > 0]
        return sum(losses) / len(losses)
    
    @property
    def run_time(self):
        return self.run_time_holder.sum
    
    @property
    def avg_run_time(self):
        return self.run_time_holder.avg
    
    @property
    def cur_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def update_tr_loss(self, dataset_idx, bs, loss, regloss=None, mlmloss=None, bs_mlm=None):
        dataset_name = self.datasets[dataset_idx].name
        self.loss_holders[dataset_name].update(loss, bs)
        if regloss is not None:
            self.reg_loss_holder.update(regloss, bs)
        if mlmloss is not None:
            self.mlm_loss_holder.update(mlmloss, bs_mlm or bs)
    
    def reset_tr_loss(self):
        """ 
        push tr loss to container and reset holder 
        execute print_loss and save_upload_results AFTER this
        """
        self.avg_train_losses.append(self.avg_loss)
        self.reg_losses.append(self.reg_loss_holder.avg)
        self.mlm_losses.append(self.mlm_loss_holder.avg)
        for name, loss_holder in self.loss_holders.items():
            self.train_losses[name].append(loss_holder.avg)
            loss_holder.reset()
        self.reg_loss_holder.reset()
        self.mlm_loss_holder.reset()
    
    @staticmethod
    def compute_loss(logits, labels, task_type, use_prompt=False):
        if use_prompt:
            return F.cross_entropy(logits, labels.view(-1), reduction='mean')
        if task_type == 'regression':
            loss = F.mse_loss(logits, labels.view(-1), reduction='mean')
        elif task_type == 'binclass':
            loss = F.binary_cross_entropy_with_logits(logits, labels.view(-1), reduction='mean')
        else:
            loss = F.cross_entropy(logits, labels.view(-1), reduction='mean')
        return loss
    
    def print_loss(self, epoch, step=None, detail=False):
        print(f"[PRINT]@Epoch[{epoch+1}/{self.max_epochs}]-Step[{step}] \
            | Avg Train Loss: {self.avg_train_losses[-1]} \
            | Avg Eval Loss: {self.avg_eval_losses[-1]} \
            | Best Metric {self.metric_policy}: {self.best_metric} \
            | Best Epoch: {self.best_epoch}, Best Step: {self.best_step} \
            | Avg Run Time: {self.avg_run_time}")
        if detail:
            for d in self.datasets:
                print(f'[PRINT] dataset: {d.name} \
                    | train loss: {self.train_losses[d.name][-1]} \
                    | eval loss: {self.eval_losses[d.name][-1]} \
                    | eval metric: {self.eval_metrics[d.name][-1]}')

    def save_and_upload_results(self, epoch, step=None, for_last=False):
        self.lrs.append(self.cur_lr)
        saved_results = {
            'status': 'training',
            'epoch': epoch + 1,
            'step': step + 1 if step is not None else None,
            f'best_{self.metric_policy}': self.best_metric,
            'best_epoch': self.best_epoch,
            'best_step': self.best_step,
            'train': {
                'avg_loss': self.avg_train_losses,
                'reg_loss': self.reg_losses,
                'mlm_loss': self.mlm_losses,
                'losses': {d.name: self.train_losses[d.name] for d in self.datasets},
            },
            'eval': {
                'avg_loss': self.avg_eval_losses,
                'metrics': {d.name: self.eval_metrics[d.name] for d in self.datasets},
                'losses': {d.name: self.eval_losses[d.name] for d in self.datasets},
            },
            'lrs': self.lrs,
            'runtime': {
                'training_time': self.run_time,
                'avg_training_time': self.avg_run_time,
                'pytorch': torch.__version__,
                'transformers': transformers.__version__,
                'gpus': {i: torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())},
            }
        }
        if for_last:
            saved_results['status'] = 'finish'
            with open(Path(self.output_dir) / 'results.json', 'w') as f:
                json.dump(saved_results, f, indent=4)
            return
        with open(Path(self.output_dir) / 'results-tmp.json', 'w') as f:
            json.dump(saved_results, f, indent=4)
        # upload to wandb
        if self.use_wandb:
            uploaded_results = {
                'avg_train_loss': saved_results['train']['avg_loss'][-1],
                'avg_reg_loss': saved_results['train']['reg_loss'][-1],
                'avg_mlm_loss': saved_results['train']['mlm_loss'][-1],
                'avg_eval_loss': saved_results['eval']['avg_loss'][-1],
                'lrs': float(self.cur_lr),
                **{f'{d.name}_tr_loss': saved_results['train']['losses'][d.name][-1] for d in self.datasets},
                **{f'{d.name}_ev_loss': saved_results['eval']['losses'][d.name][-1] for d in self.datasets},
                **{f'{d.name}_ev_metrics': saved_results['eval']['metrics'][d.name][-1] for d in self.datasets},
            }
            if step is not None:
                wandb.log(uploaded_results, step=step+1)
            else:
                wandb.log(uploaded_results, step=epoch+1)

    def save_model(self, model: Union[nn.Module, nn.DataParallel], epoch=None, step=None, for_best=False, for_last=False):
        model_dir = Path(self.output_dir) / 'pytorch_models'
        saved_model = model.module if isinstance(model, nn.DataParallel) else model
        if for_last:
            print('Saving final model')
            torch.save(saved_model.state_dict(), model_dir / f'checkpoint-{self.steps_per_epoch * self.max_epochs}.bin')
            return
        assert any([epoch, step]), 'must give epoch number or step number for saving model'
        checkpoint = f'checkpoint-{epoch+1}.bin' if step is None else f'checkpoint-{step+1}.bin'
        if for_best:
            best_suffix = 'best-epoch' if step is None else 'best-step'
            model_dir = model_dir / best_suffix
            print('Saving BEST model at: ', model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save(saved_model.state_dict(), model_dir / checkpoint)
        # check max storage
        checkpoints = [file for file in os.listdir(model_dir) if file.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x[x.index('-')+1:x.index('.')]))
        if (for_best and len(checkpoints) > self.max_best_save) \
            or (not for_best and len(checkpoints) > self.max_save):
            os.remove(model_dir / checkpoints[0])
    
    def copy_last_and_best_model(self):
        print('copying last and best model')
        model_dir = Path(self.output_dir) / 'pytorch_models'
        # for last checkpoint
        last_dir = Path(model_dir / 'last')
        os.makedirs(last_dir)
        checkpoints = [file for file in os.listdir(model_dir) if file.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x[x.index('-')+1:x.index('.')]))
        last_checkpoint = checkpoints[-1]
        shutil.copyfile(model_dir / last_checkpoint, last_dir / 'pytorch_model.bin')
        # for best checkpoint
        best_dir = Path(model_dir / 'best')
        best_suffix = 'best-epoch' if self.eval_policy == 'epoch' else 'best-step'
        model_dir = model_dir / best_suffix
        if os.path.exists(model_dir):
            os.makedirs(best_dir)
            checkpoints = [file for file in os.listdir(model_dir) if file.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x[x.index('-')+1:x.index('.')]))
            last_checkpoint = checkpoints[-1]
            shutil.copyfile(model_dir / last_checkpoint, best_dir / 'pytorch_model.bin')

            
    def evaluate(self, model, eval_datasets, epoch, step=None):
        """ 
        evaluate and push eval loss and metric to container 
        execute print_loss and save_upload_results AFTER this
        """
        device = model.module.device if isinstance(model, nn.DataParallel) else model.device
        # evaluate
        loss_holders = {d.name: AverageMeter() for d in self.datasets}
        eval_golds = {d.name: [] for d in self.datasets}
        eval_preds = {d.name: [] for d in self.datasets}
        eval_loader = MTLoader(eval_datasets, random=False)
        verbose = f'[EVAL] epoch: {epoch + 1}' + '' if step is None else f'| step: {step + 1}'
        model.eval()
        for dataset_idx, batch in tqdm(eval_loader, desc=verbose):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            bs = labels.shape[0]
            d = self.datasets[dataset_idx]
            prompt_based = d.prompt_ids is not None
            if prompt_based:
                batch['tail_prompt'] = (
                    d
                    .prompt_ids
                    .unsqueeze(0).repeat(bs, 1) # repeat before forward to avoid parellel bug, otherwise logits.shape = (b // 2)
                    .to(device)
                )
            with torch.no_grad():
                logits, _ = model(dataset_idx=dataset_idx, **batch)
                loss = self.compute_loss(logits, labels, d.task_type.value, use_prompt=prompt_based)
            loss_holders[d.name].update(loss.cpu().item(), bs)
            if not prompt_based:
                eval_golds[d.name].append(labels.cpu())
                eval_preds[d.name].append(logits.cpu())
            else:
                assert not d.is_multiclass
                if d.is_binclass:
                    label_ids = torch.tensor(d.label_enc.classes_).long()
                    eval_golds[d.name].append(torch.from_numpy(d.label_enc.transform(labels.cpu())))
                    eval_preds[d.name].append(logits[:, label_ids].softmax(-1)[:, 1].cpu()) # use positive prob
                else:
                    raise NotImplementedError('TODO: implement evaluation for prompt-based regression')

        model.train()
        # record eval results
        def output_type(d: Dataset2):
            if d.prompt_ids is not None: # prompt-based
                return 'probs' if d.is_binclass else None
            return 'logits' if not d.is_regression else None
        
        tot_loss = 0.
        for d in self.datasets:
            d_loss = loss_holders[d.name].avg
            tot_loss += d_loss
            score = calculate_metrics(
                torch.cat(eval_golds[d.name]).numpy(),
                torch.cat(eval_preds[d.name]).numpy(),
                d.task_type.value,
                output_type(d),
                d.y_info
            )
            scale = 1 if not d.is_regression else -1
            metric_key = {
                'regression': 'rmse', 
                'binclass': 'roc_auc', 
                'multiclass': 'accuracy'
            }[d.task_type.value]
            self.eval_metrics[d.name].append(scale * score[metric_key]) # push to metric container
            self.eval_losses[d.name].append(d_loss) # push to loss container
        avg_loss = tot_loss / self.n_datasets
        self.avg_eval_losses.append(avg_loss) # push to avg loss container
        # compare best metric and save if best
        if self.metric_policy == 'loss':
            if avg_loss < self.best_metric:
                self.best_metric = avg_loss
                self.best_epoch = epoch + 1
                self.best_step = step + 1
                self.save_model(model, epoch, step, for_best=True)
        else:
            raise TypeError("Undefined metric policy: ", self.metric_policy)
        # report results
        print(f'[EVAL] avg eval loss: ', avg_loss)
    
    def step_end(self, epoch, step, model: nn.Module, eval_datasets: List[DataLoader]):
        self.optimizer.step()
        self.run_time_holder.update(time.time() - self.start_time) # only calculate training time
        if (step + 1) % self.eval_steps == 0 and self.eval_policy == 'step':
            self.evaluate(model, eval_datasets, epoch, step)
            self.reset_tr_loss()
            # print and save after evaluate and reset
            if self.print_policy == 'step':
                self.print_loss(epoch, step)
            if self.save_policy == 'step':
                self.save_and_upload_results(epoch, step)
                self.save_model(model, epoch, step)
            
        self.scheduler.step()

    def step_start(self, epoch, step, model: nn.Module):
        self.optimizer.zero_grad()
        self.start_time = time.time()
    
    def epoch_end(self, epoch, model: nn.Module, eval_datasets: List[DataLoader]):
        if self.eval_policy == 'epoch':
            self.evaluate(model, eval_datasets, epoch)
            self.reset_tr_loss()
            # print and save after evaluate and reset
            # self.print_loss(epoch, detail=True)
            if self.save_policy == 'epoch':
                self.save_and_upload_results(epoch)
                self.save_model(model, epoch)
        else:
            # self.print_loss(epoch, detail=True) # print last step results if policy is 'step'
            if self.save_policy == 'epoch':
                self.save_and_upload_results(epoch)
                self.save_model(model, epoch)
    
    def epoch_start(self, epoch, model: nn.Module):
        model.train()
        print(f'[TRAIN] epoch: {epoch + 1} / {self.max_epochs}')
    
    def trainer_end(self, model: nn.Module):
        # save last model
        self.save_model(model, for_last=True)
        self.copy_last_and_best_model()
        # save all results

        # close wandb
        if self.use_wandb:
            wandb.finish()
        print('Done')
    
    def trainer_start(self, args, model: nn.Module):
        def save_run_infos():
            infos = {
                'summary': {'regression': 0, 'binclass': 0, 'multiclass': 0},
                'run_config': {
                    'optimizer': {
                        'peak_lr': self.peak_lr,
                        'min_lr': self.min_lr,
                        'warmup': self.warmup,
                        'max_epochs': self.max_epochs,
                        'lr_decay': self.lr_decay,
                        'weight_decay': self.weight_decay,
                        'eps': self.eps,
                        'beta': (self.beta1, self.beta2),
                    },
                    'training': {
                        'output_dir': self.output_dir,
                        'early_stop': self.early_stop if self.use_early_stop else -1,
                        'eval_policy': self.eval_policy,
                        'metric_policy': self.metric_policy,
                        'eval_steps': self.eval_steps,
                        'save_policy': self.save_policy,
                        'print_policy': self.print_policy,
                    }
                },
                'list': []
            }
            for d in self.datasets:
                infos['list'].append({
                    'name': d.name,
                    'task': d.task_type.value,
                    '# samples': d.size(None),
                    '# numerical features': d.n_num_features,
                    '# categorical features': d.n_cat_features,
                    '# string features': d.n_str_features,
                    '# classes': d.n_classes,
                    'metric': {
                        'regression': 'rmse', 
                        'binclass': 'auc', 
                        'multiclass': 'acc'
                    }[d.task_type.value]
                })
                infos['summary'][d.task_type.value] += 1
            # check output_dir existance (if exist raise error)
            if os.path.exists(self.output_dir):
                if not self.overwrite:
                    raise AssertionError('Existing project folder: ', self.output_dir)
                print('\nOverwrite existing project folder: ', self.output_dir)
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)
            with open(Path(self.output_dir) / 'run_infos.json', 'w') as f:
                json.dump(infos, f, indent=4)
            verbose = ' [summary]' + ' | '.join([f'{k}: {v}' for k, v in infos['summary'].items()])
            print(verbose, end='')
        
        assert self.eval_policy == self.save_policy, \
            'only allow same settings for eval_policy and save_policy'
        print("[PROMPT] begin training", end='')
        # record dataset infos
        save_run_infos()
        # start wandb
        print(' [use wandb] ' + ('True' if self.use_wandb else 'Flase'))
        if self.use_wandb:
            config = {
                # model config
                'base_model': args.bert_name,
                'max_numerical_token': args.max_numerical_token,
                'max_categorical_token': args.max_categorical_token,
                'max_feature_length': args.max_feature_length,
                'max_position_embeddings': args.max_position_embeddings,
                'type_vocab_size': args.type_vocab_size,
                # optimizer
                'peak_lr': self.peak_lr,
                'min_lr': self.min_lr,
                'warmup': self.warmup,
                'max_epochs': self.max_epochs,
                # training
                'batch_size': args.batch_size,
                'eval_policy': self.eval_policy,
                'metric_policy': self.metric_policy,
                'eval_steps': self.eval_steps,
                'early_stop': self.early_stop if self.use_early_stop else -1,
            }
            wandb.init(
                config=config,
                project='tpberta',
                entity='zju',
                name=args.run_id,
                notes=f'pre-train a {args.bert_name} model',
                job_type='training'
            )
