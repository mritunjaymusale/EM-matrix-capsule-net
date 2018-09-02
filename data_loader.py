import numpy as np
import pickle
import matplotlib.pyplot as plt


def loadFileFromPickle(filename, mode):
    filename = open(filename+".pkl", mode)
    pickle_data = pickle.load(filename)
    filename.close()
    return pickle_data


def loadTrainingSet():
    training_data = loadFileFromPickle("training", "rb")
    return training_data["data"], training_data["labels"]


def loadTestingSet():
    testing_data = loadFileFromPickle("testing", "rb")
    return testing_data["data"], testing_data["labels"]


def main():
    training_samples, training_labels = loadTrainingSet()
    testing_samples, testing_labels = loadTestingSet()


if __name__ == '__main__':
    main()
