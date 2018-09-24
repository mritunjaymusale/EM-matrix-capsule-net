import torch
import numpy as np
routing_iterations = 3
assignment_probabilities = []


def performMStep(activations):
    global assignment_probabilities
    # assignment_probabilities = assignment_probabilities * activations
    print(assignment_probabilities.shape)
    print(activations.shape)


def performEStep():
    pass


def performEMRouting(activations, votes):
    # initialize with equal probablities
    global assignment_probabilities
    assignment_probabilities = torch.zeros([32, 1])+(1/votes.shape[1])
    for t in range(routing_iterations):
        performMStep(activations)
        performEStep()
    return assignment_probabilities, 0
