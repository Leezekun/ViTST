#!/usr/bin/env bash

# PAM
for dataset_prefix in differ_interpolation_-*0.5_**1_4*5_256*320_
do
CUDA_VISIBLE_DEVICES=0 python3 run_ImgCLS.py \
    --model swin \
    --seed 1799 \
    --save_total_limit 1 \
    --dataset PAM \
    --dataset_prefix $dataset_prefix \
    --train_batch_size 48 \
    --eval_batch_size 128 \
    --logging_steps 20 \
    --save_steps 20 \
    --epochs 20 \
    --learning_rate 2e-5 \
    --n_runs 1 \
    --n_splits 5 \
    --do_train
done