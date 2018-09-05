import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional
from numpy import newaxis
from norb_loader import NORB
from torch.utils.data import DataLoader


def main():
   
    trainingdataset = DataLoader(NORB(training=True), batch_size=128)
    conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1)
    relu = nn.ReLU()

    # temporary testing
    for index, (data, labels) in enumerate(trainingdataset):
        x = conv1(data)
        x = relu(x)
        print(x)
        # print(data.shape)


if __name__ == '__main__':
    main()
