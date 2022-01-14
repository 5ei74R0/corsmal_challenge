import torch
import torch.nn.functional as F
from torch import nn


class T1Head(nn.Module):
    def __init__(self, embed_dim: int = 128, classify_num: int = 3):
        super(T1Head, self).__init__()
        self.classify_num = classify_num
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, classify_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class T2Head(nn.Module):
    def __init__(self, embed_dim: int = 128, classify_num: int = 4):
        super(T2Head, self).__init__()
        self.classify_num = classify_num
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, classify_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class T1HeadV2(nn.Module):
    def __init__(self, embed_dim: int = 128, classify_num: int = 3):
        super(T1HeadV2, self).__init__()
        self.classify_num = classify_num
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, classify_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
