#!/bin/bash -i

# Hyperparameter search for baselines
# (run to obtain tuned configs)

# regression downstream datasets
DATASETS=("HR Employee Attrition" "alerts")

for DATASET in "${DATASETS[@]}"; do
    # non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
    python scripts/finetune/tune/tune_dnns.py \
        --model "mlp" \
        --dataset "$DATASET" \
        --task "regression"

    # # FT-Transformer (using offical package)
    # python scripts/finetune/tune/tune_ftt.py \
    #     --dataset "$DATASET" \
    #     --task "regression"
    
    # # GBDTs (XGBoost, CatBoost) & tree-like DNN (TabNet)
    # python scripts/finetune/tune/tune_trees.py \
    #     --model "xgboost" \
    #     --dataset "$DATASET" \
    #     --task "regression"
done


# binclass downstream datasets
# DATASETS=("Bank_Personal_Loan_Modelling" "Churn_Modelling")

# for DATASET in "${DATASETS[@]}"; do
#     # non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
#     python scripts/finetune/tune/tune_dnns.py \
#         --model "mlp" \
#         --dataset "$DATASET" \
#         --task "binclass"
# done


# multiclass downstream datasets
# DATASETS=("xxx" "xxx")

# for DATASET in "${DATASETS[@]}"; do
#     # non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
#     python scripts/finetune/tune/tune_dnns.py \
#         --model "mlp" \
#         --dataset "$DATASET" \
#         --task "multiclass"
# done