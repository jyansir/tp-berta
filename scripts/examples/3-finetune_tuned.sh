#!/bin/bash -i

# Finetune with tuned model configs

# BEFORE RUNNING the script, you should check
# 1. hyperparameter tuning, i.e., tune.sh, should be performed to obtain tuned configs
# 2 & 3 are same as 1 & 2 in 1-finetune_default.sh

# regression downstream datasets
DATASETS=("HR Employee Attrition" "alerts")

for DATASET in "${DATASETS[@]}"; do
    # non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
    python scripts/finetune/tuned/run_tuned_config_dnns.py \
        --model "mlp" \
        --dataset "$DATASET" \
        --task "regression"

    # # FT-Transformer (using offical package)
    # python scripts/finetune/tuned/run_tuned_config_ftt.py \
    #     --dataset "$DATASET" \
    #     --task "regression"
    
    # # GBDTs (XGBoost, CatBoost) & tree-like DNN (TabNet)
    # python scripts/finetune/tuned/run_tuned_config_tree.py \
    #     --model "xgboost" \
    #     --dataset "$DATASET" \
    #     --task "regression"
done


# binclass downstream datasets
# DATASETS=("Bank_Personal_Loan_Modelling" "Churn_Modelling")

# for DATASET in "${DATASETS[@]}"; do
#     # non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
#     python scripts/finetune/tuned/run_tuned_config_dnns.py \
#         --model "mlp" \
#         --dataset "$DATASET" \
#         --task "binclass"
# done


# multiclass downstream datasets
# DATASETS=("xxx" "xxx")

# for DATASET in "${DATASETS[@]}"; do
#     # non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
#     python scripts/finetune/tuned/run_tuned_config_dnns.py \
#         --model "mlp" \
#         --dataset "$DATASET" \
#         --task "multiclass"
# done