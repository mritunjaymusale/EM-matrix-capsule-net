from torch import nn
import torch

class PrimaryCaps(nn.Module):
   

    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                              kernel_size=K, stride=stride, bias=True)
        
        self.activations = nn.Sequential(nn.Conv2d(in_channels=A, out_channels=B,
                           kernel_size=K, stride=stride, bias=True),
                           nn.Sigmoid())

    def forward(self, x):
        poses = self.pose(x)
        activations = self.activations(x)
        out = torch.cat([poses, activations], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out
