#!/bin/bash

DATASET=$1
MODEL=$2
SEED=$3
BASEDIR=$4
CONFIG=$5
GPU=$6

OUT_DIR=output/${MODEL}/${BASEDIR}/seed${SEED}_${MODEL}_${DATASET}

if [ -d $OUT_DIR ]; then
    echo "Output directory ${OUT_DIR} already exists, it will be skipped"

else
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset $DATASET \
        --out_dir $OUT_DIR \
        --model $MODEL \
        --cfg_file $CONFIG \
        --seed $SEED
fi
