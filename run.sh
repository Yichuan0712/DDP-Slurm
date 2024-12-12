#!/bin/bash

#SBATCH --gres=gpu:A100:4
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --job-name=toy_ddp
#SBATCH --output=output_ddp.log
#SBATCH --error=error_ddp.log
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -p gpu

module load miniconda3
source activate /home/yz3qt/data/miniconda/envs/splm

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

srun torchrun --nproc_per_node=4 --master_port=12345 train.py
