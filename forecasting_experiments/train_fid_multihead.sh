#!/bin/sh

export READER=t5-text
export MODELSIZE=3b
export topn=10
export trial=1

export bsz=8
export numepochs=10
export OPTIMTYPE=adamw
export weight=1e-2
export lr=5e-5

export RETR=bm25ce
export TRAIN=dataset/${RETR}_train.json
export EVAL=dataset/${RETR}_test.json
# export TRAIN=/data/andyzou_jiaming/forecasting/forecasting_data/old_data/gm_bm25_ce_all_negated_train.json
# export EVAL=/data/andyzou_jiaming/forecasting/forecasting_data/old_data/gm_bm25_ce_test.json

export DATASET=final
export TYPE=hybrid

export SCHEDULERTYPE=linear
export trainsize=4401
export warmup=100

echo ${DATASET}_${READER}-${MODELSIZE}_top${topn}_${SCHEDULERTYPE}_weight${weight}_lr${lr}_bs${bsz}_ep${numepochs}_trial${trial}_${TYPE}_retr${RETR}

export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=10000 train_reader_multihead_${TYPE}.py \
        --model_size $MODELSIZE \
        --per_gpu_batch_size $bsz \
        --epochs $numepochs \
        --answer_maxlength 10 \
        --text_maxlength 512 \
        --train_data $TRAIN \
        --eval_data $EVAL \
        --n_context $topn \
        --name ${DATASET}_${READER}-${MODELSIZE}_top${topn}_${SCHEDULERTYPE}_weight${weight}_lr${lr}_bs${bsz}_ep${numepochs}_trial${trial}_${TYPE}_retr${RETR} \
        --optim $OPTIMTYPE \
        --lr $lr \
        --weight_decay $weight \
        --scheduler $SCHEDULERTYPE \
        --warmup_steps $warmup \
        --train_data_size $trainsize \
        --use_checkpoint \

