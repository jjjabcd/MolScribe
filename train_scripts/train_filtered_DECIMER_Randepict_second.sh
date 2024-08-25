#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0

BATCH_SIZE=64
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=train_output/filtered_DECIMER_Randepict_second
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    custom_train.py \
    --data_path data \
    --train_file DECIMER/DECIMER/merged_Randepict_decimer_predictions_second_for_OCSR_train.csv \
    --valid_file DECIMER/DECIMER/filtered_DECIMER_train_val.csv \
    --vocab_file molscribe/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 256 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 2e-5 \
    --decoder_lr 2e-5 \
    --save_path $SAVE_PATH --save_mode all \
    --label_smoothing 0.1 \
    --epochs 30 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train --do_valid \
    --fp16 --backend gloo \
    --load_path ckpts/swin_base_char_aux_1m680k.pth \
    --resume 2>&1

