# Customs_ITS
Intelligent Tutoring System for Customs employee training

## Baseline model
SAINT from "Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing" based on https://arxiv.org/abs/2002.07033.  



**SAINT**: Separated Self-AttentIve Neural Knowledge Tracing. SAINT has an encoder-decoder structure where exercise and response embedding sequence separately enter the encoder and the decoder respectively, which allows to stack attention layers multiple times.  

## SAINT model architecture  
<img src="https://github.com/KAIST-AILab/Customs_ITS/blob/main/arch_from_paper.JPG">

## Usage
###Preprocessing
```shell
python preprocess.py
```
* This scripts This code combines `u*.csv` files into `train.csv` and `test.csv` files. 

###Training
```shell
python main.py --data_path {PATH_FOR_TRAIN.CSV}
               --max_seq {MAX_SEQ_LENGTH}
               --min_items {MIN_ITEMS}
               --batch_size {BATCH_SIZE}
               --lr {LEARNING_RATE}
               --h_dim {HIDDEN_DIMENSION}
               --n_layers {N_TRANSFORMER_LAYERS}
               --dropout_prob {DROPOUT_PROB}
               --epochs {EPOCHS_FOR_TRAINING}
               --gpu {GPU_INDEX}
               
```

###Testing
```shell
python test.py --data_path {PATH_FOR_TEST.CSV}
               --model_path {PATH_FOR_SAVED_MODEL}
               --max_seq {MAX_SEQ_LENGTH}
               --min_items {MIN_ITEMS}
```

## Parameters
- `h_dim`: int.  
Dimension of model ( embeddings, attention, linear layers).
- `n_layers`: int.  
Number of encoder layers, decoder layers.

## Results
* AUC: 0.5529 (original paper: 0.7811)
