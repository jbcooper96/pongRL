import torch
import torch.nn as nn
from settings import Settings

HIDDEN_SIZE = 100

class PModel(nn.Module):
    def __init__(self, action_num):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.lstm = nn.LSTM(1600, HIDDEN_SIZE, bias=True, batch_first=True)

        self.linear_out = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_num),
            nn.Softmax(dim=1)
        )

    def forward(self, state, h_0=None, c_0=None):
        batch_size = state.size()[0]
        out = self.network(state)
        out = torch.unsqueeze(out, 1)
        if h_0 == None:
            h_0 = torch.zeros(1, batch_size, HIDDEN_SIZE).to(Settings.device)
        if c_0 == None:
            c_0 = torch.zeros(1, batch_size, HIDDEN_SIZE).to(Settings.device)
        out, (h, c) = self.lstm(out, (h_0, c_0))
        out = torch.squeeze(out, 1)
        out = self.linear_out(out)
        return out, (h, c)
    
class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.lstm = nn.LSTM(1600, HIDDEN_SIZE, bias=True, batch_first=True)

        self.linear_out = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, state, h_0=None, c_0=None):
        batch_size = state.size()[0]
        out = self.network(state)
        out = torch.unsqueeze(out, 1)
        if h_0 == None:
            h_0 = torch.zeros(1, batch_size, HIDDEN_SIZE).to(Settings.device)
        if c_0 == None:
            c_0 = torch.zeros(1, batch_size, HIDDEN_SIZE).to(Settings.device)
        out, (h, c) = self.lstm(out, (h_0, c_0))
        out = torch.squeeze(out, 1)
        out = self.linear_out(out)
        return out, (h, c)

if __name__ == "__main__" :
    test = torch.rand(4, 4, 84, 84)

    action = PModel(6)
    value = ValueModel()

    actions = action(test)
    values = value(test)
    print(actions)
    print(values)