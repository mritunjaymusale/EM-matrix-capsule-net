import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional
import data_loader
from numpy import newaxis


def main():
    training_samples, training_labels = data_loader.loadTrainingSet()
    testing_samples, testing_labels = data_loader.loadTestingSet()
    training_samples = training_samples[:, :, :, newaxis]
    training_samples = torch.from_numpy(training_samples).cuda()
    conv1 = torch.nn.Conv2d(
        in_channels=1, out_channels=3, kernel_size=3, stride=1).cuda()
    
    for i in training_samples:

        x = functional.relu(conv1(i))
        print(x)


if __name__ == '__main__':
    main()
