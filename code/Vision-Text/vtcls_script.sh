#!/usr/bin/env bash

# fine-tuning the pre-trained models
CUDA_VISIBLE_DEVICES=0 python3 run_VisionTextCLS.py \
    --image_model swin \
    --text_model roberta \
    --freeze_vision_model False \
    --freeze_text_model False \
    --dataset P19 \
    --dataset_prefix order_differ_interpolation_6*6_ \
    --do_train \
    --seed 1799 \
    --save_total_limit 1 \
    --train_batch_size 48 \
    --eval_batch_size 196 \
    --logging_steps 20 \
    --save_steps 100 \
    --epochs 4 \
    --learning_rate 2e-5 \
    --n_runs 1 \
    --n_splits 5