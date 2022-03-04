'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    # def __init__(self):
    #     super(LeNet, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1   = nn.Linear(16*5*5, 120)
    #     self.fc2   = nn.Linear(120, 84)
    #     self.fc3   = nn.Linear(84, 10)
    #
    # def forward(self, x):
    #     out = F.relu(self.conv1(x))
    #     out = F.max_pool2d(out, 2)
    #     out = F.relu(self.conv2(out))
    #     out = F.max_pool2d(out, 2)
    #     out = out.view(out.size(0), -1)
    #     out = F.relu(self.fc1(out))
    #     out = F.relu(self.fc2(out))
    #     out = self.fc3(out)
    #     return out

    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        ### no bias
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(x1)
        x3 = self.conv2(x1)
        x4 = F.relu(x3)
        x5 = F.max_pool2d(x4, 2)
        x6 = self.dropout1(x5)
        x7 = torch.flatten(x6, 1)
        x8 = self.fc1(x7)
        x9 = F.relu(x8)
        x10 = self.dropout2(x9)
        x11 = self.fc2(x10)
        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11
        # return output_4


