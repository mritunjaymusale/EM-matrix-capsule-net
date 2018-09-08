from torch.utils.data import Dataset, DataLoader
import os
import torch
from torchvision import transforms
from PIL import Image


class NORB(Dataset):
    processedDirectory = "/data/smallNORB/processed/"
    training_file = 'training.pt'
    testing_file = 'test.pt'
    trainingTransformations = transforms.Compose([
        transforms.Resize(48),
        transforms.RandomCrop(32),
        transforms.ColorJitter(
            brightness=32./255, contrast=0.5),
        transforms.ToTensor()
    ])
    testingTransformations = transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])

    def __init__(self, training):
        self.trainingFlag = training
        self.loadTrainingData()
        self.loadTestingData()

    def loadTrainingData(self):
        self.train_data, self.train_labels, self.train_info = self.loadFromTorchFile(
            self.training_file)
        self.train_data, self.train_labels, self.train_info = self.performAssertionChecksAndExpandLabelset(
            self.train_data, self.train_labels, self.train_info)

    def loadTestingData(self):
        self.test_data, self.test_labels, self.test_info = self.loadFromTorchFile(
            self.testing_file)
        self.test_data, self.test_labels, self.test_info = self.performAssertionChecksAndExpandLabelset(
            self.test_data, self.test_labels, self.test_info)

    def loadFromTorchFile(self, filename):
        return torch.load(os.path.join(os.getcwd()+self.processedDirectory+filename))

    def performAssertionChecksAndExpandLabelset(self, data, labels, info):
        size = len(labels)
        assert size == len(info)
        assert size*2 == len(data)
        labels = labels.view(
            size, 1).repeat(1, 2).view(2*size, 1)
        info = info.repeat(1, 2).view(2*size, 4)
        return data, labels, info

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        if self.trainingFlag:
            img = self.performTrainingTransformations(img)
        else:
            img = self.performTestingTransformations(img)
        return img, target

    def performTrainingTransformations(self, img):
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.trainingTransformations(img)
        return img

    def performTestingTransformations(self, img):
        img = Image.fromarray(img.numpy(), mode="L")
        img = self.testingTransformations(img)
        return img

    def __len__(self):
        return len(self.train_data)


def main():
    pass


if __name__ == '__main__':
    main()
