import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from lib.saint_lib.models.pretrainmodel import SAINT as SAINTModel


class SAINT(nn.Module):
    def __init__(self, categories, d_numerical, d_out):
        super().__init__()
        if categories is None:
            categories = np.array([1]).astype(int)
        else:
            categories = np.append(np.array([1]), np.array(categories)).astype(int)

        self.model = SAINTModel(
            categories=categories,
            num_continuous=d_numerical,
            dim=32,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.8,
            attentiontype='colrow',
            y_dim=d_out
        )
    
    def embed_input(self, x_num: Tensor, x_cat: Tensor):
        device = x_num.device
        x_cat = x_cat + self.model.categories_offset.type_as(x_cat)
        x_cat_enc = self.model.embeds(x_cat)
        n1, n2 = x_num.shape
        _, n3 = x_cat.shape
        if self.model.cont_embeddings == 'MLP':
            x_cont_enc = torch.empty(n1,n2, self.model.dim)
            for i in range(self.model.num_continuous):
                x_cont_enc[:,i,:] = self.model.simple_MLP[i](x_num[:,i])
        else:
            raise Exception('This case should not work!')
        x_cont_enc = x_cont_enc.to(device)
        return x_cat_enc, x_cont_enc
    
    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]):
        b = x_num.shape[0]
        if x_cat is None:
            x_cat = torch.zeros(size=(b, 1), device=x_num.device).long()
        else:
            cls_token = torch.zeros(size=(b, 1), device=x_num.device).long()
            x_cat = torch.cat([cls_token, x_cat], dim=1)
        x_cat_enc, x_cont_enc = self.embed_input(x_num, x_cat)
        reps = self.model.transformer(x_cat_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = self.model.mlpfory(y_reps)
        return y_outs.squeeze(-1)