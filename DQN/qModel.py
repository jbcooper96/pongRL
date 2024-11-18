import torch.nn as nn
import torch

class QModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 24, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(2400, 2)
        )

    def forward(self, state):
        out = self.network(state)
        return out