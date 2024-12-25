import torch
import torch.nn as nn
from settings import Settings

class PModel(nn.Module):
    def __init__(self, action_num):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(1600, action_num),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        out = self.network(state)
        return out
    
class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(1600, 1)
        )

    def forward(self, state):
        out = self.network(state)
        return out

if __name__ == "__main__" :
    test = torch.rand(4, 4, 84, 84)

    action = PModel(6)
    value = ValueModel()

    actions = action(test)
    values = value(test)
    print(actions)
    print(values)