#!/bin/bash

#SBATCH --gres=gpu:A100:2  # 分配 2 张 A100 GPU
#SBATCH -n 1               # 单节点
#SBATCH --mem=50G
#SBATCH --job-name=toy_ddp
#SBATCH --output=output_ddp.log
#SBATCH --error=error_ddp.log
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -p gpu             # 使用 gpu 分区

module load miniconda3
source activate /home/yz3qt/data/miniconda/envs/splm

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

# 使用 srun 启动 torchrun，nproc_per_node 与 GPU 数量匹配
srun torchrun --nproc_per_node=2 --master_port=12345 train.py
