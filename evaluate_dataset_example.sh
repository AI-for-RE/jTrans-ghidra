#!/bin/bash

python3 eval_save.py \
    --model_path ./models/jTrans-finetune \
    --dataset_path DATASET_PATH \
    --experiment_path ./experiments/initial_test.pkl \
    --tokenizer ./jtrans_tokenizer/

python3 fasteval.py \
    --experiment_path ./experiments/initial_test.pkl \
    --poolsize 300