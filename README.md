# SAINT Implementation

## Baseline model
SAINT from "Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing" based on https://arxiv.org/abs/2002.07033.  



**SAINT**: Separated Self-AttentIve Neural Knowledge Tracing. SAINT has an encoder-decoder structure where exercise and response embedding sequence separately enter the encoder and the decoder respectively, which allows to stack attention layers multiple times.  

## SAINT model architecture  
<img src="https://github.com/KAIST-AILab/Customs_ITS/blob/main/arch_from_paper.JPG">

## Usage

### Preprocessing

```shell
python preprocess.py
```
* This scripts combines all `u*.csv` files and split into `train.csv`, `valid.csv` and `test.csv` files. 

### Training

```shell
python train.py --model saint\
                --data_path {PATH_FOR_TRAIN.CSV}\
                --save_path {PATH_TO_SAVE_MODEL}\
                --min_items {MIN_ITEMS}\
                --batch_size {BATCH_SIZE}\
                --lr {LEARNING_RATE}\
                --model_dim {HIDDEN_DIMENSION}\
                --n_layers {N_TRANSFORMER_LAYERS}\
                --dropout_prob {DROPOUT_PROBABILITY}\
                --seq_len {SEQUENCE_LENGTH}\
                --n_heads {N_ATTENTION_HEADS}\
                --epochs {EPOCHS_FOR_TRAINING}\
                --noam\
                --warmup_step {WARMPUP_STEP}
                --gpu {GPU_INDEX}
```
* Setting `batch_size` as 128 would give AUC similar to the one reported in paper.
* Use `--noam` option when setting `model_dim` larger than 256(256 or 512).
* Other hyper-parameters used for reproducing paper results are set as default in codes.

### Testing

```shell
python test.py --model saint\
               --data_path {PATH_FOR_TEST.CSV}\
               --model_path {PATH_FOR_SAVED_MODEL}\
               --min_items {MIN_ITEMS}\
               --model_dim {HIDDEN_DIMENSION}\
               --n_layers {N_TRANSFORMER_LAYERS}\
               --dropout_prob {DROPOUT_PROBABILITY}\
               --seq_len {SEQUENCE_LENGTH}\
               --n_heads {N_ATTENTION_HEADS}\
               --gpu {GPU_INDEX}
```
* You need to set hyper-parameters same as training.

## Results
* AUC: \
0.7723 with `model_dim`:128\
0.7714 with `model_dim`:256\
0.7689 with `model_dim`:512
* Original paper: 0.7811 with `model_dim`:512)
