#!/bin/bash -i

# Pre-train TP-BERTa from RoBerta-base

# BEFORE RUNNING the script, you should check
# 1. correctly configure PRE-TRAINING DATA PATH in lib/env.py before pre-training
# 2. the feature (column) names of all datasets have been cleaned (rerun when dataset collection is updated)

# Clean feature names
# python scripts/clean_feat_names.py --mode "pretrain" --task "binclass"

# Pre-train on binary classification datasets
python scripts/pretrain/pretrain_tpberta.py --task "binclass" --batch_size 512 --max_epochs 30

# Clean feature names
# python scripts/clean_feat_names.py --mode "pretrain" --task "regression"

# Pre-train on regression datasets
python scripts/pretrain/pretrain_tpberta.py --task "regression" --batch_size 512 --max_epochs 30


# Clean feature names
# python scripts/clean_feat_names.py --mode "pretrain" --task "binclass"
# python scripts/clean_feat_names.py --mode "pretrain" --task "regression"

# Pre-train on both binclass & regression datasets
python scripts/pretrain/pretrain_tpberta.py --task "joint" --batch_size 512 --max_epochs 30

