import torch
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)  # 生成随机特征数据
        self.labels = torch.randint(0, 2, (size,))  # 随机生成二分类标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
