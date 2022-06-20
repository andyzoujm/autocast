#!/bin/sh

export READER=t5-text
export MODELSIZE=large
export topn=1
export finetune_encoder=0
export adjust_targets=1
export trial=1

export bsz=8
export numepochs=5
export OPTIMTYPE=adamw
export weight=1e-2
export lr=5e-5

export seqlen=64

export RETR=bm25ce
export TRAIN=dataset/${RETR}_temporal_train.json
export EVAL=dataset/${RETR}_temporal_test_new.json
export DATASET=final_new

export SCHEDULERTYPE=fixed
export trainsize=4401
export warmup=100

export LOAD=final_t5-text-large_top10_linear_weight1e-2_lr5e-5_bs4_ep10_trial1_hybrid_retrbm25ce

echo ${DATASET}_${READER}-${MODELSIZE}_top${topn}_seqlen${seqlen}_${SCHEDULERTYPE}_weight${weight}_lr${lr}_ep${numepochs}_trial${trial}_retr${RETR}_finetune${finetune_encoder}_adjusttarget${adjust_targets}

python train_temporal_multihead_fast.py \
        --max_seq_len $seqlen \
        --model_size $MODELSIZE \
        --per_gpu_batch_size $bsz \
        --epochs $numepochs \
        --answer_maxlength 15 \
        --text_maxlength 512 \
        --train_data $TRAIN \
        --eval_data $EVAL \
        --n_context $topn \
        --name temporal_${DATASET}_${READER}-${MODELSIZE}_top${topn}_seqlen${seqlen}_${SCHEDULERTYPE}_weight${weight}_lr${lr}_ep${numepochs}_trial${trial}_retr${RETR}_finetune${finetune_encoder}_adjusttarget${adjust_targets} \
        --optim $OPTIMTYPE \
        --lr $lr \
        --weight_decay $weight \
        --scheduler $SCHEDULERTYPE \
        --warmup_steps $warmup \
        --train_data_size $trainsize \
        --use_checkpoint \
        --finetune_encoder $finetune_encoder \
        --adjust_targets $adjust_targets \
        --model_path checkpoint/${LOAD}/checkpoint/epoch-10
