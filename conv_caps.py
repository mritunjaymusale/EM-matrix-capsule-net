from torch import nn
import em_routing as em


kernel_size = 3
stride = 2
pose_in_channels = 512
pose_matrix = 4
pose_out_channels = pose_in_channels*kernel_size*kernel_size
activation_in_channels = 32
activation_out_channels = activation_in_channels*kernel_size*kernel_size


pose = nn.Conv2d(in_channels=pose_in_channels,
                 out_channels=pose_out_channels, kernel_size=kernel_size, stride=stride).cuda()
activation = nn.Conv2d(in_channels=activation_in_channels,
                       out_channels=activation_out_channels, kernel_size=kernel_size, stride=stride).cuda()


def performConv2d(poses, activations):
    poses = pose(poses).cuda()
    activations = activation(activations).cuda()
    return poses, activations


def reshape(tensor, dimensions):
    return tensor.view(*dimensions)


def generateVotes(poses, activations):
    # hardcoded values make them dynamic later
    votes = reshape(poses, (-1, 32*9, 4, 4))
    #TODO (size, 288, 1, 4, 4) possibly because we haven't made 32 capsules of conv layer 

    return votes, activations


def convolutionalCapsules(poses, activations):

    poses, activations = performConv2d(poses, activations)

    votes, activations = generateVotes(poses, activations)

    return poses, activations
