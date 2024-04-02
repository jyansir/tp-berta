import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from re import X
from typing import Any, Optional, Union, cast, Dict, List, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import os
import json
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from . import env, util
from .util import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=True)
class Dataset2:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    X_str: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    feature_names: Dict[str, List[str]]
    task_type: TaskType
    n_classes: Optional[int]
    name: Optional[str]
    prompt_infos: Optional[Dict[str, str]] = None
    prompt_ids: Optional[torch.Tensor] = None
    label_enc: Optional[LabelEncoder] = None

    @classmethod
    def from_csv(
        cls, 
        data_dir: Union[Path, str], 
        data_name: str, 
        max_cat_num: Optional[int] = None, 
        no_str: bool = False, 
        min_y_frequency: Union[int, float] = 2,
        tt: str = None, # task type
    ) -> 'Dataset2':

        csv_file = Path(data_dir) / f'{data_name}.csv'

        df = pd.read_csv(csv_file)
        max_cat_num = max_cat_num or int(len(df) / 100)

        # treat the last column as the label column in default
        feat_types = [str(df.iloc[:, i].dtype) for i in range(df.shape[1]-1)]
        feat_names = np.array(df.columns[:-1])

        # convert categorical features into strings
        def cat2str(li: Dict[int, str]):
            li = {int(k): v for k, v in li.items()}
            enc = LabelEncoder()
            max_idx = max(li.keys())
            missing_cnt = 0
            classes_ = []
            for i in range(max_idx+1):
                if i not in li:
                    classes_.append(f'none{missing_cnt}')
                    missing_cnt += 1
                else:
                    classes_.append(li[i])
            enc.classes_ = np.array(classes_)
            return enc
        
        # categorical feature value infos (e.g., [feature: color] 0 -> 'red, 1 -> 'green')
        if os.path.exists(data_dir / 'cat_infos.json'):
            with open(data_dir / 'cat_infos.json', 'r') as f:
                cat_infos = json.load(f).get(data_name, None)
        else:
            cat_infos = None

        # label infos (e.g., [label name] 0 -> 'false', 1 -> 'true')
        # for prompt-based tuning (not used in this paper)
        if os.path.exists(data_dir / 'prompt_infos.json'):
            with open(data_dir / 'prompt_infos.json', 'r') as f:
                prompt_infos = json.load(f).get(data_name, None)
        else:
            prompt_infos = None

        # map standard feature name (e.g., [feature name] BloodPressure -> 'blood pressure')
        with open(data_dir / 'feature_names.json', 'r') as f:
            feature_map = json.load(f)

        X_num_idx, X_cat_idx, X_str_idx = [], [], []
        for i, ft in enumerate(feat_types):
            feature_words = feature_map[feat_names[i]].split(' ')
            # if 'ID' in feature name
            if any('id' == w.lower() for w in feature_words):
                if len(df.iloc[:, i].unique()) == len(df):
                    pass # useless ID column (# IDs == # samples)
                elif len(df.iloc[:, i].unique()) <= 100 or ft == 'object':
                    if no_str:
                        X_cat_idx.append(i) # non-LM baselines
                    else:
                        X_str_idx.append(i) # LM baselines
                else:
                    X_num_idx.append(i)
                continue
            
            # accept string format (LM) & feature value type is object
            if not no_str and ft == 'object':
                X_str_idx.append(i)

            # feature with limited unique values
            elif len(df.iloc[:, i].unique()) <= max_cat_num:
                # numerical features
                if ft == 'float64':
                    uniques = df.iloc[:, i].unique()
                    n_unique = len(uniques)
                    if any(df[df.columns[i]].isna()):
                        n_unique -= 1
                    if all(x in uniques for x in range(n_unique)) and not no_str:
                        X_str_idx.append(i)
                    else:
                        X_num_idx.append(i)
                    continue
                # categorical features with feature value map
                elif cat_infos is not None and feat_names[i] in cat_infos:
                    li = cat_infos[feat_names[i]]
                    if isinstance(li, dict) and not no_str:
                        X_str_idx.append(i)
                        enc = cat2str(li)
                        df.iloc[:, i] = enc.inverse_transform(df.iloc[:, i])
                    elif li == 'num':
                        X_num_idx.append(i)
                    else:
                        X_cat_idx.append(i)
                # no feature value map
                else:
                    if not no_str: # for LMs
                        X_str_idx.append(i)
                    else:        
                        X_cat_idx.append(i)
            # feature with too many unique values
            else:
                if ft == 'object': # if string values
                    # don't support string input that exceed max category number
                    # X_cat_idx.append(i)
                    continue
                X_num_idx.append(i) # numerical values
        
        X_num, X_cat, X_str = None, None, None
        # convert categorical features into string features (if string is short)
        feature_names = {'num': [], 'cat': [], 'str': []}
        if len(X_num_idx) > 0:
            feature_names['num'] = feat_names[X_num_idx].tolist()
            X_num = {'train': df.iloc[:, X_num_idx].values.astype(np.float32)}
        if len(X_cat_idx) > 0: # for non-LM baselines
            feature_names['cat'] = feat_names[X_cat_idx].tolist()
            X_cat = {'train': df.iloc[:, X_cat_idx].values.astype(np.object_)}
        if len(X_str_idx) > 0: # for LM baselines (e.g., TP-BERTa), all categorical features are treated as strings
            feature_names['str'] = feat_names[X_str_idx].tolist()
            X_str = {'train': df.iloc[:, X_str_idx].values.astype(np.str).astype(np.object_)}
        

        def remove_data(rm_mask):
            if any(rm_mask):
                ys['train'] = ys['train'][~rm_mask]
                if X_num is not None:
                    X_num['train'] = X_num['train'][~rm_mask]
                if X_cat is not None:
                    X_cat['train'] = X_cat['train'][~rm_mask]
                if X_str is not None:
                    X_str['train'] = X_str['train'][~rm_mask]
        
        # y info (we assume the csv file contains labels)
        # scripts for data file without labels should be modified here
        task_type = tt
        assert task_type in ['binclass', 'regression', 'multiclass']
        
        # process y
        ys = {'train': df.iloc[:, -1].values} # label is the last column
        # check nan in y
        if ys['train'].dtype == np.object_:
            remove_mask = ys['train'].astype(str) == 'nan'
        else:
            remove_mask = np.isnan(ys['train'])
        remove_data(remove_mask)
    
        
        # check rare y for multiclass
        remove_ys = {}
        if task_type == 'multiclass':
            y_counter = Counter(ys['train'])
            if min_y_frequency < 1:
                min_y_frequency = int(len(ys['train']) * min_y_frequency)
            remove_ys = {k for k, v in y_counter.items() if v <= min_y_frequency}
            remove_mask = np.array([False] * len(ys['train']))
            for rare_y in remove_ys:
                remove_mask |= (ys['train'] == rare_y)
            remove_data(remove_mask)

     
        # encode string ys into number
        if ys['train'].dtype == np.object_:
            if task_type == 'regression':
                ys['train'] = ys['train'].astype(np.float32)
            else:
                assert task_type in ['multiclass', 'binclass']
                pass # will encode in the subsequent pipeline

        # check multiclass dataset y start from 0 to |C|-1
        if task_type == 'multiclass':
            if ys['train'].dtype == np.object:
                ys['train'] = LabelEncoder().fit_transform(ys['train'])
            if ys['train'].dtype != np.int64:
                ys['train'] = ys['train'].astype(np.int64)
            if set(ys['train']) != set(range(len(np.unique(ys['train'])))):
                enc = LabelEncoder()
                enc.classes_ = np.array(sorted(set(ys['train'])))
                ys['train'] = enc.transform(ys['train'])
            
        n_classes = len(np.unique(ys['train'])) if task_type == 'multiclass' else 1
        if task_type == 'regression':
            ys = {k: v.astype(np.float32) for k, v in ys.items()}

        return Dataset2(
            X_num,
            X_cat,
            X_str,
            ys,
            {},
            feature_names,
            TaskType(task_type),
            n_classes,
            data_name,
            prompt_infos,
        )
    
    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]
    
    @property
    def n_str_features(self) -> int:
        return 0 if self.X_str is None else self.X_str['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features + self.n_str_features

    def size(self, part: Optional[str]) -> int:
        if part == 'test' and 'test' not in self.y:
            for x in ['X_num', 'X_cat', 'X_str']:
                data = getattr(self, x)
                if data is not None:
                    return len(data['test'])
            return 0
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])


def num_process_nans(dataset: Dataset2, policy: Optional[NumNanPolicy]) -> Dataset2:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset


# Inspired by: https://github.com/Yura52/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int]
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        noise = 1e-3
        if noise > 0:
            assert seed is not None
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()} # can't perform elementwise comparison if v is not 'str/object' type
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
) -> Tuple[ArrayDict, bool]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X['train'])
    X = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X['train'].max(axis=0)
    for part in ['val', 'test']:
        if part not in X:
            continue
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # Step 2. Encode.
    if encoding is None:
        return (X, False)
    elif encoding == 'one-hot':
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32  # type: ignore[code]
        )
        encoder.fit(X['train'])
        return ({k: encoder.transform(v) for k, v in X.items()}, True)  # type: ignore[code]
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        encoder = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
        return (X, True)  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
        else:
            if y['train'].dtype == 'object':
                y = {k: LabelEncoder().fit_transform(v) for k, v in y.items()}
    else:
        util.raise_unknown('policy', policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


def transform_dataset(
    dataset: Dataset2,
    transformations: Transformations,
    cache_dir: Optional[Path],
) -> Dataset2:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{dataset.name}__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    X_num = dataset.X_num
    if dataset.X_cat is None:
        replace(transformations, cat_nan_policy=None, cat_min_frequency=None, cat_encoding=None)
        # assert transformations.cat_nan_policy is None
        # assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    if X_num is not None and transformations.normalization is not None:
        X_num = normalize(X_num, transformations.normalization, transformations.seed)

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    return dataset


def prepare_tensors(
    dataset: Dataset2, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    if isinstance(device, str):
        device = torch.device(device)
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y


def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = env.DATA / dataset_dir_name
    info = util.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info
