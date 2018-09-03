import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional
import data_loader
from numpy import newaxis


class Convolution(nn.Module):
    def __init__(self, ):
        super(Convolution, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=5, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


def main():
    training_samples, training_labels = data_loader.loadTrainingSet()
    testing_samples, testing_labels = data_loader.loadTestingSet()
    # training_samples = training_samples[:, newaxis, :, :]
    training_samples = torch.from_numpy(training_samples).cuda()
  


if __name__ == '__main__':
    main()
