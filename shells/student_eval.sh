#!/bin/bash
set -e

device=0
seed=42
dataset_key="T15"
teacher_model="Qwen_2.5_VL_72B"

CUDA_VISIBLE_DEVICES=${device} python scripts/student_eval.py \
    --dataset_key $dataset_key \
    --student_model_size 3B \
    --test_batch_size 24 \
    --seed 42 \
    --generate_max_length 512\
    --teacher_model ${teacher_model} \
    --lr 0.0003 | tee logs/step3_${dataset_key}_${teacher_model}_eval.txt