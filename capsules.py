from torch import nn
from torch.utils.data import DataLoader
from norb_loader import NORB
import primary_caps as pc
import conv_caps as cc


trainingdataset = DataLoader(NORB(training=True), batch_size=128)

# ReluConv layer
conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
relu = nn.ReLU()


def main():

    # temporary testing
    # every iteration of for loop will receive 128 samples
    for index, (data, labels) in enumerate(trainingdataset, 0):
        x = conv1(data)
        x = relu(x)
        pose, activation = pc.primaryCapsules(x)
        pose, activation = cc.convolutionalCapsules(pose, activation)


if __name__ == '__main__':
    main()
