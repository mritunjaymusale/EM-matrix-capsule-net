import numpy as np
import torch
from torch import nn
from torch.nn import functional



class ConvolutionLayer(nn.Module):

    def __init__(self):
        super(ConvolutionLayer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=5, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = functional.relu(x)
        return x


class Capsules (nn.Module):

    def __init__(self):
        super(Capsules, self).__init__()
        self.poseMatrix = torch.zeros(4, 4)
        self.activationProbability = torch.tensor([0])


class Votes (nn.Module):

    def __init__(self):
        super(Votes, self).__init__()
        self.transformationMatrix = torch.zeros(4, 4)
        self.voteMatrix = torch.zeros(4, 4)


