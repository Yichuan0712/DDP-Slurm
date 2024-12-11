import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 简单的全连接层：输入 10 维，输出 2 类

    def forward(self, x):
        return self.fc(x)
