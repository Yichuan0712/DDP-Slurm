
---

### **2. scripts/ddp_toy.slurm**
```bash
#!/bin/bash
#SBATCH --job-name=toy_ddp           # 作业名称
#SBATCH --output=output_ddp.log      # 标准输出
#SBATCH --error=error_ddp.log        # 错误输出
#SBATCH --gres=gpu:A100:4            # 分配 4 张 A100 GPU
#SBATCH --mem=50G                    # 内存
#SBATCH --cpus-per-task=8            # 每任务使用 8 个 CPU
#SBATCH --time=00:30:00              # 最长运行时间
#SBATCH -p gpu                       # 使用 GPU 分区

# 加载模块和环境
module load miniconda3
source activate /home/yz3qt/data/miniconda/envs/splm

# 运行分布式训练
torchrun --nproc_per_node=4 --master_port=12345 src/train.py
