from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import NumBins

def magnitude_regloss(b: int, num_bins: NumBins, _model: Union[nn.Module, nn.DataParallel]):
    """auxiliary regularization: magnitude-aware triplet loss"""
    model = _model.module if isinstance(_model, nn.DataParallel) else _model
    embeddings = model.base_model.get_input_embeddings()
    ranker = model.ranker # projector for relative magnitude token rank
    device = embeddings.weight.device

    start, end = num_bins.start_token_id, num_bins.start_token_id + num_bins.max_count
    # Triplet Ranking Loss
    points = np.random.randint(low=start, high=end, size=(b, 3))
    points.sort(-1)
    points = torch.from_numpy(points).to(device)
    anchors = points[:, 1] # middle value, (b,)
    left, right = points[:, 0], points[:, 2]
    right_is_pos = (2 * anchors - left - right) > 0
    pos = right * right_is_pos + left * ~right_is_pos # (b,)
    neg = right * ~right_is_pos + left * right_is_pos # (b,)
    # the difference between bin distances (normalized)
    neg_dis_minus_pos_dis = (torch.abs(anchors - neg) - torch.abs(anchors - pos)).float() / num_bins.max_count # (b,)
    m = neg_dis_minus_pos_dis.detach()
    # fetch embeddings of each bin
    anchors = embeddings(anchors)
    pos = embeddings(pos)
    neg = embeddings(neg)
    # projection
    anchors = ranker(anchors)
    pos = ranker(pos)
    neg = ranker(neg)
    # triplet loss with L1 distance
    def distance(x, y):
        return torch.linalg.norm(x - y, dim=1)
    loss = m + distance(anchors, pos) - distance(anchors, neg)
    loss_mask = loss > 0
    loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    return loss

