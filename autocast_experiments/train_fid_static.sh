#!/bin/sh

export READER=t5
export MODELSIZE=3b
export TOPN=10

export BSZ=8
export EPOCHS=10
export OPTIMTYPE=adamw
export SCHEDULERTYPE=linear
export WDECAY=1e-2
export LR=5e-5
export TRAINSIZE=4401 # number of train examples
export WARMUP=100

export RETR=bm25ce
export TRAIN=dataset/${RETR}_static_train.json
export EVAL=dataset/${RETR}_static_test.json

export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=10000 train_fid_static.py \
        --model_size $MODELSIZE \
        --per_gpu_batch_size $BSZ \
        --epochs $EPOCHS \
        --answer_maxlength 10 \
        --text_maxlength 512 \
        --train_data $TRAIN \
        --eval_data $EVAL \
        --n_context $TOPN \
        --name ${READER}_${MODELSIZE}_top${TOPN}_${SCHEDULERTYPE}_wdecay${WDECAY}_lr${LR}_bs${BSZ}_ep${EPOCHS}_retr${RETR} \
        --optim $OPTIMTYPE \
        --lr $LR \
        --weight_decay $WDECAY \
        --scheduler $SCHEDULERTYPE \
        --warmup_steps $WARMUP \
        --train_data_size $TRAINSIZE \
        --use_checkpoint \
