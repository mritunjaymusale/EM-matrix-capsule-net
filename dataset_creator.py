import numpy as np
from scipy.misc import imread
import os

traindatasetdirectory = "./smallNORB/train/"
testdatasetdirectory = "./smallNORB/test/"


def properPathCreator(filepath):
    files = os.listdir(filepath)
    files = [os.path.join(filepath, i)for i in files]
    return files


def makeTrainingDataset(datasetdirectory):
    files = properPathCreator(datasetdirectory)
    traindataset = np.asarray([imread(i) for i in files])
    return traindataset


def makeTestingDataset(datasetdirectory):
    files = properPathCreator(datasetdirectory)
    testdataset = np.asarray([imread(i) for i in files])
    return testdataset

#TODO save them in a numpy zip file 
makeTrainingDataset(traindatasetdirectory)
makeTestingDataset(testdatasetdirectory)
