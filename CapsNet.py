from ConvCaps import ConvCaps
import torch.nn as nn

from PrimaryCaps import PrimaryCaps
from ClassCaps import ClassCaps


class CapsNet(nn.Module):
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3, cuda=True):
        super(CapsNet, self).__init__()
        self.cuda = cuda
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                  momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(
            B, C, K, P, stride=2, iters=iters, cuda=self.cuda)
        self.conv_caps2 = ConvCaps(
            C, D, K, P, stride=1, iters=iters, cuda=self.cuda)
        self.class_caps = ClassCaps(D, E, 1, P, stride=1, iters=iters,
                                    coor_add=True, w_shared=True, cuda=self.cuda)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.class_caps(x)
        return x


def capsules(**kwargs):
    model = CapsNet(**kwargs)
    return model
