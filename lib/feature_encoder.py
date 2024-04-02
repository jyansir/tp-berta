from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import trange
import numpy as np

from typing import Optional, Dict, Any
from dataclasses import dataclass, replace
from .data import Dataset2

@dataclass
class NumBins:
    """
    Binner for numerical values
    ---
    Reference
    https://github.com/yandex-research/rtdl-num-embeddings/blob/bb7bb675f20664fc72b63fec072f4ece22fceaab/bin/train1.py
    """
    start_token_id: int = -1 # the first extra token id for relative magnitude tokens, assigned as tokenizer.mask_token_id + 1 in default
    max_count: int = 128 # max token number for RMT / max bin number for numerical feature discretization
    min_samples_bin: int = 8 # min sample number in each leaf node / bin

    def discrete_num(self, dataset: Dataset2):
        """C4.5 discretization (default)"""
        if dataset.X_num is None:
            return dataset

        bin_edges = []
        _bins = {x: [] for x in dataset.X_num}

        for feature_idx in trange(dataset.n_num_features):
            train_column = dataset.X_num['train'][:, feature_idx]
            # build tree model
            tree = (DecisionTreeRegressor if dataset.is_regression else DecisionTreeClassifier)(
                max_leaf_nodes=self.max_count, min_samples_leaf=self.min_samples_bin
            ).fit(train_column.reshape(-1,1), dataset.y['train']).tree_

            tree_thresholds = []
            for node_id in range(tree.node_count):
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    tree_thresholds.append(tree.threshold[node_id])
            tree_thresholds.append(train_column.max())
            tree_thresholds.append(train_column.min())
            bin_edges.append(np.array(sorted(set(tree_thresholds))))

            for spl in dataset.X_num:
                _bins[spl].append(
                    np.digitize(
                        dataset.X_num[spl][:, feature_idx],
                        np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf],
                    ).astype(np.int64)
                    - 1 + self.start_token_id
                )
        
        _bins = {k: np.stack(v, axis=1) for k, v in _bins.items()}
        return replace(dataset, X_num=_bins)

@dataclass
class CatEncoder:
    """Encoder for categorical features"""
    start_token_id: int = 0
    max_cat_features: int = 30 # the max number of categorical features [0, N-1]
    
    def encode_cat(self, dataset: Dataset2):
        if dataset.X_cat is None:
            return dataset
        X_cat = dataset.X_cat
        if X_cat['train'].dtype != np.int64:
            X_cat = {k: v.astype(np.int64) for k, v in X_cat.items()}
        for spl in X_cat:
            for feature_idx in range(dataset.n_cat_features):
                n_unique = len(set(X_cat[spl][:, feature_idx]))
                if n_unique == 2:
                    # binary feature
                    X_cat[spl][:, feature_idx] += self.start_token_id
                else:
                    assert n_unique <= self.max_cat_features, f'unique value number for \
                        categorical feature is {self.max_cat_features}, but {n_unique} for feature #{feature_idx}'
                    # categorical features
                    X_cat[spl][:, feature_idx] += self.start_token_id + 2
        return replace(dataset, X_cat=X_cat)
