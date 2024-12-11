import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset import ToyDataset
from model import ToyModel

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 加载模型
    model = ToyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 加载数据集
    dataset = ToyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(5):
        sampler.set_epoch(epoch)  # 确保每个 epoch 的数据随机性
        for data, labels in dataloader:
            data, labels = data.to(local_rank), labels.to(local_rank)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed on GPU {local_rank}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
