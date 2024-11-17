import torch
import torch.nn as nn
import random

INPUT_SIZE = 210 * 160 
HIDDEN_SIZE = 2000
OUTPUT_SIZE = 2

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=8, stride=2, padding=3), #105 80
            nn.LeakyReLU(),
            nn.Conv2d(24, 24, kernel_size=4, stride=2, padding=1), #52 40
            nn.LeakyReLU(),
            nn.Conv2d(24, 24, kernel_size=4, stride=2), #26 20
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(11400, 2),
            nn.Softmax(dim=0)
        )
        """
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Sigmoid(),
            nn.Softmax(dim=0)
        )
        """

    def forward(self, obs):
       
        obs = torch.unsqueeze(obs, dim=0)
        up_prob = self.layers(obs)
        return up_prob