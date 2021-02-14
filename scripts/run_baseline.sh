#!/bin/sh

EPOCH=200
BATCH_SIZE=64

LR=0.1
LR_DECAY_STEP_SIZE=60
LR_DECAY_GAMMA=0.1
WEIGHT_DECAY=0.0001

python src/main.py\
        --epoch=${EPOCH}\
        --batch-size=${BATCH_SIZE}\
        --lr=${LR}\
        --weight-decay=${WEIGHT_DECAY}\
        --lr-decay-step-size=${LR_DECAY_STEP_SIZE}\
        --lr-decay-gamma=${LR_DECAY_GAMMA}\
        --amp\
        --contain-test




