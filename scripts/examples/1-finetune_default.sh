#!/bin/bash -i

# Finetune with default model configs

# BEFORE RUNNING the script, you should check
# 1. correctly configure FINETUNE DATA PATH in lib/env.py before fine-tuning the downstream datasets
# 2. similar to 0-pretrain.sh, feature name clean should be performed first for LM-based model
# python scripts/clean_feat_names.py --mode "finetune" --task "binclass"
# python scripts/clean_feat_names.py --mode "finetune" --task "regression"
# python scripts/clean_feat_names.py --mode "finetune" --task "multiclass"


# regression downstream datasets
DATASETS=("HR Employee Attrition" "alerts")

for DATASET in "${DATASETS[@]}"; do
    # TP-BERTa
    python scripts/finetune/default/run_default_config_tpberta.py \
        --dataset "$DATASET" \
        --task "regression"
    
    # # GBDTs (XGBoost, CatBoost)
    # python scripts/finetune/default/run_default_config_tree.py \
    #     --model "xgboost" \
    #     --dataset "$DATASET" \
    #     --task "regression"
    
    # # TabNet
    # python scripts/finetune/default/run_default_config_tabnet.py \
    #     --dataset "$DATASET" \
    #     --task "regression"
    
    # # TransTab
    # python scripts/finetune/default/run_default_config_transtab.py \
    #     --dataset "$DATASET" \
    #     --task "regression"
    
    # # XTab (should download its weights before)
    # python scripts/finetune/default/run_default_config_xtab.py \
    #     --num_pretrain_rounds "1k" \
    #     --dataset "$DATASET" \
    #     --task "regression"
done


# binclass downstream datasets
# DATASETS=("Bank_Personal_Loan_Modelling" "Churn_Modelling")

# for DATASET in "${DATASETS[@]}"; do
#     # TP-BERTa
#     python scripts/finetune/default/run_default_config_tpberta.py \
#         --dataset "$DATASET" \
#         --task "binclass"
# done


# multiclass downstream datasets
# DATASETS=("xxx" "xxx")

# for DATASET in "${DATASETS[@]}"; do
#     # TP-BERTa
#     python scripts/finetune/default/run_default_config_tpberta.py \
#         --dataset "$DATASET" \
#         --task "multiclass"
# done