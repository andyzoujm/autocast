#!/bin/sh

export READER=t5
export MODELSIZE=3b
export TOPN=2

export seqlen=128 # maximum lookback days
export finetune_encoder=0 # freeze FiD Static model
export adjust_targets=1 # improve crowd forecasts with true resolution

export BSZ=8
export EPOCHS=5
export OPTIMTYPE=adamw
export SCHEDULERTYPE=fixed
export WDECAY=1e-2
export LR=5e-5
export TRAINSIZE=4387 # number of train examples
export WARMUP=100

export RETR=bm25ce
export TEST_QUESTIONS=data/test_questions.json
export TRAIN_QUESTIONS=data/train_questions.json
export TRAIN_CROWD=data/train_crowd.json
export TEST_CROWD=data/test_crowd.json
export TRAIN_SCHEDULE=data/train_schedule.json
export TEST_SCHEDULE=data/test_schedule.json

# export LOAD=t5_${MODELSIZE}_top10_linear_wdecay1e-2_lr5e-5_bs8_ep10_retrbm25ce # load FiD Static model

python train_fid_temporal.py \
        --max_seq_len $seqlen \
        --finetune_encoder $finetune_encoder \
        --adjust_targets $adjust_targets \
        --model_size $MODELSIZE \
        --per_gpu_batch_size $BSZ \
        --epochs $EPOCHS \
        --answer_maxlength 15 \
        --text_maxlength 512 \
        --train_questions $TRAIN_QUESTIONS \
        --test_questions $TEST_QUESTIONS \
        --train_crowd $TRAIN_CROWD \
        --test_crowd $TEST_CROWD \
        --train_schedule $TRAIN_SCHEDULE \
        --test_schedule $TEST_SCHEDULE \
        --n_context $TOPN \
        --name temporal_${READER}_${MODELSIZE}_top${TOPN}_seqlen${seqlen}_${SCHEDULERTYPE}_wdecay${WDECAY}_lr${LR}_bs${BSZ}_ep${EPOCHS}_retr${RETR}_finetune${finetune_encoder}_adjusttarget${adjust_targets} \
        --optim $OPTIMTYPE \
        --lr $LR \
        --weight_decay $WDECAY \
        --scheduler $SCHEDULERTYPE \
        --warmup_steps $WARMUP \
        --train_data_size $TRAINSIZE \
        --use_checkpoint \
        --model_path checkpoint/${LOAD}/checkpoint/epoch-10
