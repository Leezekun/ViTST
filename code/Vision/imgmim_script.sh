#!/usr/bin/env bash
# scripts for runing masked image modeling

# fine-tuning the pre-trained models
CUDA_VISIBLE_DEVICES=0 python3 run_ImgMIM.py \
    --model swin \
    --do_train \
    --seed 88 \
    --save_total_limit 1 \
    --dataset P19 \
    --dataset_prefix order_differ_interpolation_6*6_ \
    --train_batch_size 48 \
    --eval_batch_size 192 \
    --logging_steps 10 \
    --save_steps 20 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --n_runs 1 \
    --n_splits 5 \
    --mask_patch_size 32 \
    --mask_ratio 0.5 \
    --mask_method vertical