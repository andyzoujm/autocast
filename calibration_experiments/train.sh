#!/bin/sh

export T=all_train_std.csv # Log normalized
export V=all_test_std.csv
export SCRIPT=finetune_confidence

export ML=128
export BS=100
export E=5
export MODEL=large

python finetune_confidence.py \
  --model_name_or_path microsoft/deberta-v3-${MODEL} \
  --seed 0 \
  --max_length $ML \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size 50 \
  --learning_rate 2e-5 \
  --num_train_epochs $E \
  --output_dir output \
  --train_file data/${T} \
  --validation_file data/${V} \
  --output_dir output/${MODEL}
