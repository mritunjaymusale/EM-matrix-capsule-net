import torch

routing_iterations = 3
assignment_probabilities = []


def performMStep():

    pass


def performEStep():
    pass


def performEMRouting(activations, votes):
    # initialize with equal probablities
    global assignment_probabilities
    assignment_probabilities = torch.zeros([32, 1])+(1/votes.shape[1])
    for t in range(routing_iterations):
        performMStep()
        performEStep()
    return assignment_probabilities, 0
