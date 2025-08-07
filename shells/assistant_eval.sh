#!/bin/bash
set -e

seed=42
device=0
dataset_key="T15"
teacher_model="Qwen_2.5_VL_72B"

CUDA_VISIBLE_DEVICES=${device} python scripts/assistant_eval.py \
    --dataset_key ${dataset_key} \
    --teacher_model ${teacher_model} \
    --test_batch_size 16 \
    --generate_max_length 512 \
    --ft_lr 3e-4 \
    --seed ${seed} | tee logs/step1_${dataset_key}_${teacher_model}_eval.txt