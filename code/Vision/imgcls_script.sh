#!/usr/bin/env bash

# fine-tuning the pre-trained models
CUDA_VISIBLE_DEVICES=1 python3 run_ImgCLS.py \
    --model swin \
    --do_train \
    --seed 88 \
    --save_total_limit 1 \
    --dataset PAM \
    --dataset_prefix differ_interpolation_4*5_ \
    --train_batch_size 72 \
    --eval_batch_size 196 \
    --logging_steps 20 \
    --save_steps 20 \
    --epochs 20 \
    --learning_rate 2e-5 \
    --n_runs 1 \
    --n_splits 5