import torch
from torch import nn
import em_routing as em
import numpy as np

kernel_size = 3
stride = 2
pose_in_channels = 512
pose_matrix = 4
pose_out_channels = pose_in_channels*kernel_size*kernel_size
activation_in_channels = 32
activation_out_channels = activation_in_channels*kernel_size*kernel_size


transformation_matrix = None
assignment_probabilities = None
initial_single_probability = None

pose = nn.Conv2d(in_channels=pose_in_channels,
                 out_channels=pose_out_channels, kernel_size=kernel_size, stride=stride)
activation = nn.Conv2d(in_channels=activation_in_channels,
                       out_channels=activation_out_channels, kernel_size=kernel_size, stride=stride)


def calculateVotes(pose_matrix):
    votes_matrix = pose_matrix*transformation_matrix
    return votes_matrix


def createTransformationMatrix(poses):
    global transformation_matrix
    transformation_matrix = torch.rand(poses.shape, requires_grad=True)


def performConv2d(poses, activations):
    poses = pose(poses)
    activations = activation(activations)
    return poses, activations


def reshapePoses(poses):
    # hardcoded values 9and 512 find a way to make dynamic later
    poses = poses.view(poses.shape[0], 512, 9, poses.shape[2], poses.shape[3])
    return poses


def reshapeActivations(activations):
    activations = activations.view(
        activations.shape[0], 32, 9, activations.shape[2], activations.shape[3])
    return activations


def convolutionalCapsules(poses, activations):
    poses, activations = performConv2d(poses, activations)
    poses = reshapePoses(poses)
    activations = reshapeActivations(activations)
    # createTransformationMatrix(poses)
    # votes = calculateVotes(poses)
    # new_activations, new_poses = em.performEMRouting(activations, votes)

    # return new_activations, new_poses
