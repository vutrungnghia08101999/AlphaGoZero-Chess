import matplotlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=104, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.batchNorm = nn.BatchNorm2d(num_features=256)

    def forward(self, s):
        assert s.shape[1] == 8 and s.shape[2] == 8 and s.shape[3] == 104
        s = s.view(-1, 104, 8, 8)
        s = F.relu(self.batchNorm(self.conv(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d2 = nn.BatchNorm2d(num_features=256)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.BatchNorm2d1(out))
        out = self.conv2(out)
        out = self.BatchNorm2d2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # policy block
        self.conv1 = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=1)
        self.batchNorm2d1 = nn.BatchNorm2d(2)
        # self.dropOut = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * 8 * 2, 8 * 8 * 78)

        # value block
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1, stride=1)  # value head
        self.batchNorm2d2 = nn.BatchNorm2d(1)
        self.fc2 = nn.Linear(8 * 8 * 1, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s):
        # policy block
        p = F.relu(self.batchNorm2d1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 2)
        p = self.fc1(p)
        p = F.softmax(p, dim=1)

        # value block
        v = F.relu(self.batchNorm2d2(self.conv2(s)))
        v = v.view(-1, 8 * 8 * 1)  # batch_size X channel X height X width
        v = F.relu(self.fc2(v))
        v = torch.tanh(self.fc3(v))
        return p, v

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        p, v = self.outblock(s)
        return p, v
