import numpy as np
from scipy.misc import imread
import os
import pickle


traindatasetdirectory = "./smallNORB/train/"
testdatasetdirectory = "./smallNORB/test/"


def filenameExtrator(filepath):
    filenames = os.listdir(filepath)
    return filenames


def labelExtractor(filenames):
    labels = [i.split("_")[1] for i in filenames]
    return labels


def properPathCreator(filepath):
    filenames = filenameExtrator(filepath)
    filenamesWithDirectory = [os.path.join(filepath, i)for i in filenames]
    return filenamesWithDirectory


def makeTrainingDataset(datasetdirectory):
    files = properPathCreator(datasetdirectory)
    traindataset = np.asarray([imread(i) for i in files])
    filenames = filenameExtrator(datasetdirectory)
    labels = labelExtractor(filenames)
    return traindataset, labels


def makeTestingDataset(datasetdirectory):
    files = properPathCreator(datasetdirectory)
    testdataset = np.asarray([imread(i) for i in files])
    filenames = filenameExtrator(datasetdirectory)
    labels = labelExtractor(filenames)
    return testdataset, labels


def saveFile(dataset, filename):
    savefile = open(filename+'.pkl', 'wb')
    pickle.dump(dataset, savefile)
    savefile.close()


def main():
    train_data, train_labels = makeTrainingDataset(traindatasetdirectory)
    test_data, test_labels = makeTestingDataset(testdatasetdirectory)
    train_dataset = {"data": train_data, "labels": train_labels}
    test_dataset = {"data": test_data, "labels": test_labels}
    saveFile(train_dataset, filename="training")
    saveFile(test_dataset, filename="testing")


main()
