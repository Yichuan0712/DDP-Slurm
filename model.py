import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)
