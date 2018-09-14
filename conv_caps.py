import torch
from torch import nn
import em_routing as em
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


def calculateVotes(pose_matrix):
    votes_matrix = pose_matrix*transformation_matrix
    return votes_matrix


def createTransformationMatrix(poses):

    return torch.rand(poses.shape, requires_grad=True)


def convolutionalCapsules(poses, activations):
    # can't put this in calculateVotes() since it will reinitailize everytime the func is called
    global transformation_matrix
    transformation_matrix = createTransformationMatrix(poses)
    
    votes = calculateVotes(poses)
    new_activations, new_poses = em.performEMRouting(activations, votes)
    return new_activations, new_poses
