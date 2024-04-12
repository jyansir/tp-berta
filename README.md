# TP-BERTa: A Fundamental LM Adaption Technique to Tabular Data

This repo contains original PyTorch implementation of:

- [Making Pre-trained Language Models Great on Tabular Prediction](https://openreview.net/pdf?id=anzIzGZuLi) (ICLR 2024 Spotlight)

## Key Features

The following key features are proposed in this paper:

- Relative magnitude tokenization (*RMT*): a distributed representation method for continuous values to enhance LM's numerical perception capability.

- Intra-feature attention (*IFA*): a mechanism to pre-fuse feature-wise information for reasonable tabular feature contexts & model acceleration.

- *TP-BERTa*: a resulting LM pre-trained from RoBERTa with the above features for tabular prediction.

## Project Structure

The repo structure and module functions are as follows:

```
├─bin ---- // Implementation of tabular models
│ ├─tpberta_modeling.py ---- // TP-BERTa base class
│ └─xxx.py ----------------- // Other non-LM DNN baselines
├─lib ---- // Utilities
│ ├─aux.py --------------- // Auxiliary Loss: Magnitude-aware Triplet Loss
│ ├─feature_encoder.py --- // Numerical Value Binner (C4.5 discretization)
│ ├─optim.py ------------- // Utilities for optimizer & trainer
│ ├─env.py --------------- // Environment Variables configs
│ ├─data.py -------------- // Dataset & Data Transformation class
│ ├─data_utils.py -------- // Data Config & Multi-task Loader class
│ └─xxx.py --------------- // Other standard utils
├─data --- // csv file path for pre-training & fine-tuning
│ ├─pretrain-bin
│ ├─pretrain-reg
│ ├─finetune-bin
│ ├─finetune-reg
│ └─finetune-mul
├─checkpoints --- // Pre-trained model weights & configs (RoBERTa, TP-BERTa)
├─configs --- // Model & Training configs for non-LM baselines
│ ├─default --- // default configs
│ └─tuned ----- // tuned configs (generated with hyperparameter tuning scripts)
├─scripts --- // Experiment codes
│ ├─examples --- // Example shell scripts for main experiments
│ ├─pretrain --- // Codes for TP-BERTa pre-training
│ ├─finetune --- // Codes for baseline fine-tuning & hyperparameter tuning
│ └─clean_feat_names.py --- // Text clean for table feature names
```

## Dependencies

All necessary dependencies for TP-BERTa are included in `requirement.txt`. To conduct the packaged baselines, uncomment the corresponding lines.

### How to pre-train a TP-BERTa from scratch

In experiment we saved weights and configs of [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base/tree/main) in the local `checkpoints/roberta-base` folder (network unavailable) and conducted pre-training with `scripts/pretrain/pretrain_tpberta.py`. You can use online HuggingFace APIs by assigning the argument `--base_model_dir` with "FacebookAI/roberta-base".

## TODO

- [x] Upload pre-trained TP-BERTa checkpoints.
    1. Download TP-BERTa checkpoints pre-trained on [single task type](https://drive.google.com/uc?export=download&id=13_GAK2VcShxm5TgqSvLk2afBTIYcCbEs) or [both task types](https://drive.google.com/uc?export=download&id=1ArjkOAblGPErmxUyVIfpiM0IztnjjYxq).
    2. Move the `*.tar.gz` file to the `checkpoints` folder (create one if not exists)
    3. Unzip the file and run TP-BERTa according to the scripts in `scripts/examples/finetune`.

- [x] Sort and update experiment datasets.
    1. We have acquired permission on distributing the used data subset from [TabPertNet (OpenTabs currently)](https://arxiv.org/abs/2307.04308) datasets.
    2. Download datasets for [pre-training](https://drive.google.com/uc?export=download&id=1Jy45I_vTKn6McMROi5IKjKoSi9QJtx9A) (202 datasets) and [fine-tuning](https://drive.google.com/uc?export=download&id=1JhOJR1kxjyu4w4ZHi8VcxgMh-iYJRDgG) (145 datasets).
    3. Unzip the `*.tar.gz` file to the `data` folder (create one if not exists).

- [ ] Integrate TP-BERTa to HuggingFace🤗 community.


## Citation

If you find this useful for your research, please cite the following paper:

```
@article{yan2024making,
  title={Making Pre-trained Language Models Great on Tabular Prediction},
  author={Yan, Jiahuan and Zheng, Bo and Xu, Hongxia and Zhu, Yiheng and Chen, Danny and Sun, Jimeng and Wu, Jian and Chen, Jintai},
  journal={arXiv preprint arXiv:2403.01841},
  year={2024}
}
```

## Acknowledgments

Our codes are influenced by the following repos:

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [RTDL Numerical Embeddings](https://github.com/yandex-research/rtdl-num-embeddings)