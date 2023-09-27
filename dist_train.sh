#!/usr/bin/env bash

# 第0个参数是脚本名
CONFIG=$1 # 命令行第1个参数，配置文件名 
GPUS=$2 # 命令行第2个参数，使用的GPU数量

# ${variable:-DefaultValue} 如果变量variable没有被声明，或者为空，则返回DefaultValue
NNODES=${NNODES:-1} # 使用的计算节点数，mmdetection只支持单机多卡，所以计算节点为1
NODE_RANK=${NODE_RANK:-0} # 计算节点的rank，如果为 0 代表是 master 节点（机器）
PORT=${PORT:-29501} # master 节点开放的端口号
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} # master 节点的 IP 地址

torchrun --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --launcher pytorch ${@:3} # 其中$@表示命令行中所有参数，且把每个参数区分对待。 而${@:3}表示从第3个参数开始，将剩余的参数传递给train.py


# 单机多卡启动多个作业示例：
# CUDA_VISIBLE_DEVICES = 0,1,2,3 PORT = 29500 ./dist_train.sh $ {CONFIG_FILE} 4
# CUDA_VISIBLE_DEVICES = 4,5,6,7 PORT = 29501 ./dist_train.sh $ {CONFIG_FILE} 4