from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import replace, dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import json
import copy
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

from .env import DATA
from .feature_encoder import NumBins, CatEncoder
from .data import Dataset2, Transformations, transform_dataset, TensorDict, CAT_MISSING_VALUE

@dataclass(frozen=True)
class DataConfig:
    num_cont_token: int
    num_cat_token: int
    num_bin_token: int = 2
    max_class_num: int = 30
    max_feature_length: int = 100
    max_seq_length: int = 512
    train_ratio: float = 0.8
    pre_train: bool = False # For pre-training or not
    prompt_based: bool = False
    # base_model: str = 'roberta-base'
    tokenizer: Optional[RobertaTokenizer] = None
    num_encoder: Optional[NumBins] = None
    cat_encoder: Optional[CatEncoder] = None # TODO: rm further
    feature_map: Optional[Dict[str, str]] = None
    feature_map_file: Union[Path, str] = None
    data_dir: Optional[Path] = None
    use_cache: bool = True
    # training data config
    batch_size: int = 1024
    # val_batch_size: int = None
    preproc_type: str = None
    # numerical embeddings strategy
    use_num_multiply: bool = False

    @classmethod
    def from_default(cls, args, **kwargs):
        default_config = {
            'num_cont_token': args.max_numerical_token,
            'num_cat_token': args.max_categorical_token,
            'max_class_num': 30,
            'max_feature_length': args.max_feature_length,
            'max_seq_length': args.max_seq_length,
            'train_ratio': 0.8,
            'tokenizer_dir': Path(args.base_model_dir),
            'data_dir': None, # need to be overwritten in kwargs
            'feature_map_file': args.feature_map,
            'batch_size': args.batch_size,
            **kwargs,
            # 'batch_size': 32,
            # 'val_batch_size': 128
        }
        assert 'preproc_type' in default_config
        assert default_config['preproc_type'] in ['lm', 'ftt', 'lm_all_str']
        return DataConfig.from_config(default_config)
    
    def save_pretrained(self, result_dir):
        saved_config = {
            'num_cont_token': self.num_cont_token,
            'num_cat_token': self.num_cat_token,
            'max_class_num': self.max_class_num,
            'max_feature_length': self.max_feature_length,
            'max_seq_length': self.max_seq_length,
            'pre_train': self.pre_train,
            'train_ratio': self.train_ratio,
            'data_dir': str(self.data_dir),
            'feature_map_file': os.path.basename(str(self.feature_map_file)),
            'batch_size': self.batch_size,
        }
        with open(Path(result_dir) / 'data_config.json', 'w') as f:
            json.dump(saved_config, f, indent=4)
        self.tokenizer.save_pretrained(result_dir)
        print('save data config at: ', result_dir)

    @classmethod
    def from_pretrained(cls, pretrain_dir, **kwargs):
        print("load data config from:", pretrain_dir)
        with open(Path(pretrain_dir) / 'data_config.json', 'r') as f:
            config = json.load(f)
        config.update(kwargs)
        config['data_dir'] = Path(config['data_dir'])
        tokenizer = RobertaTokenizer.from_pretrained(pretrain_dir)
        return DataConfig.from_config(config, tokenizer)

    @classmethod
    def from_config(cls, config: dict, tokenizer=None):
        tokenizer_dir = config.pop('tokenizer_dir', None)
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir)
            tokenizer.add_tokens([f'nbin{i}' for i in range(config['num_cont_token'])])
            tokenizer.add_tokens([f'bbin{i}' for i in range(2)])
            tokenizer.add_tokens([f'cbin{i}' for i in range(config['num_cat_token'])])
        # numerical encoder
        num_encoder = NumBins(
            start_token_id=tokenizer.added_tokens_encoder['nbin0'],
            max_count=config['num_cont_token']
        )
        # categorical encoder
        cat_encoder = CatEncoder(
            start_token_id=tokenizer.added_tokens_encoder['bbin0'],
            max_cat_features=config['num_cat_token']
        )
        # standard feature name map
        feature_map_file = config.get('feature_map_file')
        data_dir = config.get('data_dir', Path(DATA))
        with open(data_dir  /feature_map_file, 'r') as f:
            feature_map = json.load(f)
        return DataConfig(**config, tokenizer=tokenizer, num_encoder=num_encoder, cat_encoder=cat_encoder, feature_map=feature_map)
    
    def clone(self, **kwargs):
        return replace(copy.copy(self), **kwargs)


class TensorDictDataset(Dataset[TensorDict]):
    tensor_dict: TensorDict

    def __init__(self, tensor_dict: TensorDict) -> None:
        super().__init__()
        assert all(tensor_dict['input_ids'].size(0) == tensor.size(0) for tensor in tensor_dict.values())
        self.tensor_dict = tensor_dict
    
    def __getitem__(self, index):
        return {k: tensor[index] for k, tensor in self.tensor_dict.items()}
    
    def __len__(self):
        return self.tensor_dict['input_ids'].size(0)


@dataclass
class MTLoader:
    """A big data loader container to incorperate multiple datasets loader in one"""
    dataloaders: List[DataLoader]
    random: bool = True
    test: bool = False

    def __len__(self) -> int:
        return sum([len(dl) for dl in self.dataloaders])
    
    @property
    def n_datasets(self):
        return len(self.dataloaders)
    
    def __getitem__(self, index):
        return self.dataloaders[index]
    
    def __iter__(self):
        # self.n_remain_datasets = len(self.dataloaders)
        self.dataloaders = [iter(dl) for dl in self.dataloaders]
        self.remain_idx = list(range(len(self.dataloaders)))
        self.cur_idx = 0
        return self
    
    def __next__(self):
        if self.random:
            while True:
                if len(self.remain_idx) == 0:
                    raise StopIteration
                i = np.random.choice(self.remain_idx)
                try:
                    batch = next(self[i])
                except StopIteration:
                    batch = None
                    self.remain_idx.remove(i)
                if batch is None:
                    continue
                return i, batch
        elif self.test:
            while True: # traverse dataset loop
                if len(self.remain_idx) == 0:
                    raise StopIteration
                if self.cur_idx == len(self.remain_idx):
                    self.cur_idx = 0 # return first dataset
                i = self.remain_idx[self.cur_idx]
                try:
                    batch = next(self[i])
                    self.cur_idx += 1
                except StopIteration:
                    batch = None
                    self.remain_idx.remove(i)
                if batch is None:
                    continue
                return i, batch
        else:
            while True:
                if self.cur_idx == self.n_datasets:
                    raise StopIteration
                try:
                    batch = next(self[self.cur_idx])
                except StopIteration:
                    batch = None
                    self.cur_idx += 1
                if batch is None:
                    continue
                return self.cur_idx, batch

def split(dataset: Dataset2, config: DataConfig = None):
    if config and config.train_ratio == 1.0: # no eval / test
        return dataset
    if all(spl in dataset.y for spl in ['val', 'test']): # eval / test already exists
        return dataset
    
    new_datas = {}
    data_pieces = ['y', 'X_num', 'X_cat', 'X_str']
    train_ratio = config.train_ratio if config is not None else 0.8 # default: 0.8
    test_ratio = 1 - train_ratio
    stratify = None if dataset.is_regression else dataset.y['train']
    sample_idx = np.arange(len(dataset.y['train']))
    train_idx, val_idx = train_test_split(sample_idx, test_size=test_ratio, random_state=42, stratify=stratify)
    
    if config and config.pre_train or 'test' in dataset.y:
        test_idx = None # no need for test split
    else:
        test_idx = val_idx
        stratify = None if dataset.is_regression else dataset.y['train'][train_idx]
        train_idx, val_idx = train_test_split(train_idx, test_size=test_ratio, random_state=42, stratify=stratify)

    for piece in data_pieces:
        data = getattr(dataset, piece)
        if data is None:
            new_datas[piece] = None
        else:
            new_datas[piece] = {}
            new_datas[piece]['train'] = data['train'][train_idx]
            new_datas[piece]['val'] = data['train'][val_idx]
            if test_idx is None:
                if 'test' not in data: # test `y` not given
                    continue
                new_datas[piece]['test'] = data['test']
            else:
                new_datas[piece]['test'] = data['train'][test_idx]
    return replace(dataset, **new_datas)


def data_preproc(dataset_name, config: DataConfig = None, no_str=False, tt:str = None) -> Dataset2:
    assert os.path.exists(config.data_dir / f'{dataset_name}.csv'), f'DATASET [{dataset_name}] not exists'
    if config is not None:
        dataset = Dataset2.from_csv(config.data_dir, dataset_name, max_cat_num=config.num_cat_token, no_str=no_str, tt=tt)
    else:
        # default data reading
        dataset = Dataset2.from_csv(DATA, dataset_name, max_cat_num=16, no_str=no_str, tt=tt)
    
    """ Data Transformation """
    # `ftt`: non-LM dnn baselines like FT-Transformer (FTT)
    # `LM`: LM baselines like TP-BERTa
    # `xgboost`, `catboost`, ...
    if config.preproc_type in ['ftt', 'lm']:
        normalization = 'quantile' # standard DNN normalization
        num_nan_policy = (
            'mean' if dataset.X_num is not None \
                and any(np.isnan(dataset.X_num[spl]).any() \
                        for spl in dataset.X_num) else None
        ) # numerical feature is not None & NaN values exist
        cat_nan_policy = (
            'most_frequent' if dataset.X_cat is not None \
                and any((dataset.X_cat[spl] == CAT_MISSING_VALUE).any() \
                        for spl in dataset.X_cat) else None
        ) # categorical feature is not None & NaN values exist
        cat_min_frequency = 0.05 if dataset.X_cat is not None else None # same as FTT settings
        transformation = Transformations(normalization=normalization, num_nan_policy=num_nan_policy, cat_nan_policy=cat_nan_policy, cat_min_frequency=cat_min_frequency)
    elif config.preproc_type in ['xgboost']:
        num_nan_policy = 'mean' if dataset.X_num is not None and any(np.isnan(dataset.X_num[spl]).any() for spl in dataset.X_num) else None
        transformation = Transformations(normalization=None, num_nan_policy=num_nan_policy, cat_nan_policy=None, cat_min_frequency=None, cat_encoding='one-hot')
    elif config.preproc_type in ['catboost']:
        num_nan_policy = 'mean' if dataset.X_num is not None and any(np.isnan(dataset.X_num[spl]).any() for spl in dataset.X_num) else None
        transformation = Transformations(normalization=None, num_nan_policy=num_nan_policy, cat_nan_policy=None, cat_min_frequency=None, cat_encoding=None)
    else:
        raise NotImplementedError('Please add new preproc_type for your baseline')
    
    # [INFO] using cache will overwrite later modification in data split 
    # or new changes of this dataset, change to None or delete the cache
    cache_dir = (
        None if (config is None or not config.use_cache)
        else config.data_dir / 'cache' / config.preproc_type
    ) # save cache or not, if you want to use new split, set to None
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    prompt_infos = dataset.prompt_infos # protect prompt infos for `transform_dataset` will read cache to overwrite it
    dataset = split(dataset, config) # split into train, val, test
    dataset = transform_dataset(dataset, transformation, cache_dir) # WARNING: cache will overwrite data split
    dataset = replace(dataset, prompt_infos=prompt_infos)
    return dataset


def encode_single_dataset(
    dataset: Dataset2, 
    config: DataConfig, 
):
    """ Data Tokenization for TP-BERTa """
    num_biner, cat_encoder, tokenizer, feature_name_map = \
        config.num_encoder, config.cat_encoder, config.tokenizer, config.feature_map

    ori_X_num = dataset.X_num
    # numerical feature discretization
    dataset = num_biner.discrete_num(dataset)
    # categorical feature encoding
    dataset = cat_encoder.encode_cat(dataset)

    feature_names = dataset.feature_names
    all(spl in feature_names for spl in ['num', 'cat', 'str'])
    feature_names = {spl: [feature_name_map[v] for v in feature_names[spl]] for spl in ['num', 'cat', 'str']}
    encoded_feature_names = {
        spl: [
            tokenizer.encode(v, add_special_tokens=False)[:config.max_feature_length] 
            for v in feature_names[spl]
        ] for spl in ['num', 'cat', 'str']}

    # prepare encoded pieces
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    # label infos for prompt-based tuning 
    # (NOT USED prompt-based tuning in this paper, only using [CLS] for prediction)
    # these codes can be used for further exploration
    # prompt_length = 0 # add to outputs after IntraFeatureAttention
    # prompt_ids = []
    # label_words = []
    # if config.prompt_based and dataset.y is not None: # None y for test file without labels
    #     assert not dataset.is_multiclass, 'prompt-based tunning not support multiclass yet, you can split it into N binclass tasks'
    #     assert dataset.prompt_infos is not None, 'perform prompt-based tunning needs label infos'
    #     prompt = f"label <{dataset.prompt_infos['label_name']}> is {tokenizer.pad_token}" # add pad token manually
    #     prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    #     prompt_length = len(prompt_ids)
    #     dataset = replace(dataset, prompt_ids=torch.tensor(prompt_ids))
    #     # multiclass can be decomposed into N binclass thus we don't consider multiclass in pre-training
    #     label_words = (
    #         ['false', 'true']
    #         if dataset.is_binclass
    #         else [f'reg{i}' for i in range(128)] # reg label bin: catb based
    #         if dataset.is_regression
    #         else None
    #     )
    #     label_words = [tokenizer.encode(lw, add_special_tokens=False)[0] for lw in label_words]
    #     label_enc = LabelEncoder()
    #     label_enc.classes_ = np.array(label_words)
    #     # prompt-based tunning label
    #     if config.prompt_based:
    #         new_y = {}
    #         for spl in dataset.y:
    #             new_y[spl] = label_enc.inverse_transform(dataset.y[spl])
    #     dataset = replace(dataset, prompt_ids=torch.tensor(prompt_ids), label_enc=label_enc, y=new_y)

    encoded = {}
    def has_spl(spl):
        return any(
            spl in x for x in [
                dataset.y, dataset.X_num, 
                dataset.X_str, dataset.X_cat
            ] if x is not None
        )
    
    for spl in ['train', 'val', 'test']:
        if not has_spl(spl):
            continue
        N = dataset.size(spl)

        # numerical features
        num_fix_part = [] if dataset.n_num_features > 0 else None
        num_token_types = [] if dataset.n_num_features > 0 else None
        num_position_ids = [] if dataset.n_num_features > 0 else None
        num_feature_cls_mask = [] if dataset.n_num_features > 0 else None
        num_input_scales = [] if dataset.n_num_features > 0 else None
        num_num_token = 0
        offset = 1
        if num_fix_part is not None:
            for i, efn in enumerate(encoded_feature_names['num']):
                num_fix_part.extend([cls_token_id] + efn + [mask_token_id])
                num_token_types.extend([0] + [0] * len(efn) + [1]) # continous type (1)
                num_feature_cls_mask.extend([1] + [0] * len(efn) + [0])
                num_position_ids.extend([0] + [i for i in range(1, len(efn)+1)] + [0])
                if config.use_num_multiply:
                    num_input_scales.append(ori_X_num[spl][:, i:i+1].repeat(1 + len(efn), axis=1))
                    
            num_fix_part = np.repeat(np.array([num_fix_part]), repeats=N, axis=0)
            num_fix_part[num_fix_part == mask_token_id] = dataset.X_num[spl].reshape(-1)
            num_num_token = num_fix_part.shape[1]
            num_token_types = np.repeat([num_token_types], repeats=N, axis=0)
            num_position_ids = np.repeat([num_position_ids], repeats=N, axis=0)
            num_feature_cls_mask = np.repeat([num_feature_cls_mask], repeats=N, axis=0)
            if not config.use_num_multiply:
                num_input_scales = np.ones_like(num_token_types, dtype=np.float32)
                num_input_scales[num_token_types == 1] = ori_X_num[spl].reshape(-1)
            else:
                num_input_scales = np.concatenate(num_input_scales, axis=1)
        
        # discrete features
        cat_fix_part = [] if dataset.n_cat_features > 0 else None
        cat_token_types = [] if dataset.n_cat_features > 0 else None
        cat_position_ids = [] if dataset.n_cat_features > 0 else None
        cat_feature_cls_mask = [] if dataset.n_cat_features > 0 else None
        cat_input_scales = [] if dataset.n_cat_features > 0 else None
        num_cat_token = 0
        offset += dataset.n_num_features
        if cat_fix_part is not None:
            for i in range(dataset.n_cat_features):
                cat_features = dataset.X_cat[spl][:, i]
                assert cat_features.dtype == np.int64, f'categorical feature should be long data type'
                unique_cats = set(cat_features)
                # DEBUG: promise each categorical feature appears in training set ?
                # assert unique_cats == set(range(len(unique_cats))), f'feature_idx #{i} in "{spl}" split should be in range [0,N-1]'
                efn = encoded_feature_names['cat'][i]
                each_part = np.repeat(np.array([[cls_token_id] + efn + [mask_token_id]]), repeats=N, axis=0)

                # input ids
                each_part[each_part == tokenizer.mask_token_id] = cat_features

                # token type: binary type  (2), categorical type (3)
                feature_type = 2 if len(unique_cats) == 2 else 3
                each_token_type = np.repeat(np.array([[0] + [0] * len(efn) + [feature_type]]), repeats=N, axis=0)

                # position ids
                each_position_ids = np.repeat(np.array([[0] + [i for i in range(1, len(efn)+1)] + [0]]), repeats=N, axis=0)

                # feature cls mask
                each_feature_mask = np.repeat(np.array([[1] + [0] * len(efn) + [0]]), repeats=N, axis=0)

                cat_fix_part.append(each_part)
                cat_token_types.append(each_token_type)
                cat_position_ids.append(each_position_ids)
                cat_feature_cls_mask.append(each_feature_mask)
            cat_fix_part = np.concatenate(cat_fix_part, axis=1)
            cat_token_types = np.concatenate(cat_token_types, axis=1)
            cat_position_ids = np.concatenate(cat_position_ids, axis=1)
            cat_feature_cls_mask = np.concatenate(cat_feature_cls_mask, axis=1)
            cat_input_scales = np.ones_like(cat_token_types, dtype=np.float32)
            num_cat_token = cat_fix_part.shape[1]
        
        # string features
        str_part = [[] for _ in range(N)]
        str_token_types = [[] for _ in range(N)]
        str_position_ids = [[] for _ in range(N)]
        str_feature_cls_mask = [[] for _ in range(N)]
        offset += dataset.n_cat_features
        if str_part is not None:
            for i in range(dataset.n_str_features):
                str_features = dataset.X_str[spl][:, i]
                assert str_features.dtype == np.object_
                unique_strs = set(str_features)
                # hash map for quick process
                tmp_map = {s: tokenizer.encode(s, add_special_tokens=False) for s in unique_strs}
                efn = encoded_feature_names['str'][i]
                for j in range(N):
                    esv = tmp_map[str_features[j]] # encoded string value
                    # input ids
                    str_part[j].extend([cls_token_id] + efn + esv) # feature name + specific string value
                    # token types: string type (4)
                    str_token_types[j].extend([0] + [0] * len(efn) + [4] * len(esv))
                    # position ids
                    str_position_ids[j].extend([0] + [i for i in range(1, len(efn)+1)] + [0] * len(esv))
                    # feature cls mask
                    str_feature_cls_mask[j].extend([1] + [0] * len(efn) + [0] * len(esv))
        
        # padding to max length
        num_fix_part_token = num_num_token + num_cat_token
        max_str_tokens = config.max_seq_length - num_fix_part_token - 2 # 2: [cls], [sep]
        max_str_tokens = min(max([len(str_tokens) for str_tokens in str_part]), max_str_tokens)
        offset += dataset.n_str_features
        for i in range(N):
            # add [sep] and pad str part token
            pad_length = max_str_tokens - len(str_part[i]) # remaining length to pad
            if pad_length < 0: # exceed max length
                str_part[i] = str_part[i][:pad_length]
                str_token_types[i] = str_token_types[i][:pad_length]
                str_position_ids[i] = str_position_ids[i][:pad_length]
                str_feature_cls_mask[i] = str_feature_cls_mask[i][:pad_length]
            str_part[i] += [sep_token_id] + [pad_token_id] * max(0, pad_length) # [x] [sep] [pad] ...
            str_token_types[i] +=  [0] + [0] * pad_length
            str_position_ids[i] += [0] + [0] * pad_length
            str_feature_cls_mask[i] += [0] + [0] * pad_length
        str_input_scales = np.ones_like(str_token_types, dtype=np.float32)
        
        str_part = np.array(str_part)
        str_token_types = np.array(str_token_types)
        str_position_ids = np.array(str_position_ids)
        # add [cls] and concate all parts
        input_ids = np.concatenate(
            [np.ones((N,1), dtype=int) * cls_token_id] 
            + [token for token in [num_fix_part, cat_fix_part, str_part] 
            if token is not None], axis=1)
        token_type_ids = np.concatenate(
            [np.zeros((N,1), dtype=int)] 
            + [token_type for token_type in [num_token_types, cat_token_types, str_token_types] 
            if token_type is not None], axis=1)
        position_ids = np.concatenate(
            [np.zeros((N,1), dtype=int)] 
            + [pos_ids for pos_ids in [num_position_ids, cat_position_ids, str_position_ids] 
            if pos_ids is not None], axis=1)
        feature_cls_mask = np.concatenate(
            [np.ones((N,1), dtype=int)] 
            + [x for x in [num_feature_cls_mask, cat_feature_cls_mask, str_feature_cls_mask] 
            if x is not None], axis=1)
        input_scales = np.concatenate(
            [np.ones((N,1), dtype=np.float32)] 
            + [x for x in [num_input_scales, cat_input_scales, str_input_scales] 
            if x is not None], axis=1)

        encoded[spl] = {
            'input_ids': input_ids, 
            'input_scales': input_scales.astype(np.float32), 
            'feature_cls_mask': feature_cls_mask, 
            'token_type_ids': token_type_ids, 
            'position_ids': position_ids,
        }
        if spl in dataset.y:
            encoded[spl]['labels'] = dataset.y[spl]
    
    return encoded, dataset


def prepare_tpberta_loaders(
    _datasets: List[str], 
    data_config: Union[DataConfig, Dict[str, DataConfig]], 
    n_datasets:int = 0, 
    tt: Optional[Union[str, List[str]]] = None,
) -> Tuple[List[DataLoader], List[Dataset2]]:
    """ Prepare dataloaders for a list of datasets"""
    n_datasets = min(n_datasets, len(_datasets)) if n_datasets > 0 else len(_datasets)

    datasets = [] # all dataset objects
    data_loaders = [] # all dataloaders
    for i in range(n_datasets):
        print(f'\r prepare datasets #({i+1}/{n_datasets}) [{_datasets[i]}]', end='')

        dataset = data_preproc(_datasets[i], data_config, tt=tt)
        encoded, dataset = encode_single_dataset(dataset, data_config)
        datasets.append(dataset)
        
        task_type = dataset.task_type.value
        assert task_type == tt
        if task_type == 'multiclass':
            task_type += str(dataset.n_classes)
        enc_dataset = {}
        for spl in encoded:    
            if 'labels' in encoded[spl]:
                if task_type.startswith('multiclass'): # multiclass label
                    encoded[spl]['labels'] = encoded[spl]['labels'].astype(np.int64)
                else:
                    encoded[spl]['labels'] = encoded[spl]['labels'].astype(np.float32)
            encoded[spl] = {k: torch.as_tensor(v) for k, v in encoded[spl].items()}
            enc_dataset[spl] = TensorDictDataset(encoded[spl])
        
        data_loaders.append((
            {k: DataLoader(
                    dataset=v,
                    batch_size=data_config.batch_size,
                    shuffle=k == 'train',
                ) for k, v in enc_dataset.items()},
            task_type))
    return data_loaders, datasets
