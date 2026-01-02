#!/usr/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1
script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

# 先根据 CUDA_VISIBLE_DEVICES 判断用几张卡
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NGPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
else
    # 没设就默认全卡
    NGPU=$(nvidia-smi --list-gpus | wc -l)
fi

if [ -z "$WORLD_SIZE" ]; then
    echo "Training on 1 Node, $NGPU GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
    torchrun --nnodes=1 \
        --master_port=29906 \
        --nproc_per_node=$NGPU \
        --node_rank=0 \
        $script_path \
        --config_file $config_path
else
echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
NGPU=`nvidia-smi --list-gpus | wc -l`
torchrun --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NGPU \
    --node_rank=$RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path
fi
