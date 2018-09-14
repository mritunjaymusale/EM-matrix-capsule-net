import torch
from torch import nn

kernel_size = 3
stride = 2
pose_in_channels = 32
pose_matrix = 4
pose_out_channels = 32
activation_in_channels = 32
activation_out_channels = 32

transformation_matrix = None
assignment_probabilities = None
initial_single_probability = None


def calculateVotes(pose_matrix, transformation_matrix):

    votes_matrix = pose_matrix*transformation_matrix
    return votes_matrix


def performEMRouting(activatons, votes):
    # initialize with equal probablities
    assignment_probabilities = torch.zeros([32, 1])+(1/votes.shape[1])


def convolutionalCapsules(poses, activations):
    # can't put this in calculateVotes() since it will reinitailize everytime the func is called
    transformation_matrix = torch.rand(poses.shape, requires_grad=True)

    votes = calculateVotes(poses, transformation_matrix)
    new_activations, new_poses = performEMRouting(activations, votes)
