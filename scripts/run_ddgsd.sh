#!/bin/sh

EPOCH=200
BATCH_SIZE=64

LR=0.1
LR_DECAY_STEP_SIZE=60
LR_DECAY_GAMMA=0.1
WEIGHT_DECAY=0.0001

LAMBDA=1.0
MU=0.0005
TEMPERATURE=3.0

python src/main.py\
        --epoch=${EPOCH}\
        --batch-size=${BATCH_SIZE}\
        --lr=${LR}\
        --weight-decay=${WEIGHT_DECAY}\
        --lr-decay-step-size=${LR_DECAY_STEP_SIZE}\
        --lr-decay-gamma=${LR_DECAY_GAMMA}\
        --lmbda=${LAMBDA}\
        --mu=${MU}\
        --temperature=${TEMPERATURE}\
        --ddgsd\
        --amp\
        --contain-test

