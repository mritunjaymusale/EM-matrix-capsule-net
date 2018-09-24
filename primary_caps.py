from torch import nn

kernel_size = 1
stride = 1
pose_in_channels = 32
pose_matrix = 4
pose_out_channels = 32*pose_matrix*pose_matrix
activation_in_channels = 32
activation_out_channels = 32

pose = nn.Conv2d(in_channels=pose_in_channels,
                 out_channels=pose_out_channels, kernel_size=kernel_size, stride=stride)
activation = nn.Sequential(nn.Conv2d(
    in_channels=activation_in_channels, out_channels=activation_out_channels, kernel_size=kernel_size, stride=stride),
    nn.Sigmoid()
)


def primaryCapsules(current_input):
    transformed_poses = pose(current_input)
    transformed_activation = activation(current_input)
    return transformed_poses, transformed_activation
