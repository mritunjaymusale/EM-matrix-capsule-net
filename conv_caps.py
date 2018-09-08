import torch
from torch import nn

kernel_size = 1
stride = 1
pose_in_channels = 32
pose_matrix = 4
pose_out_channels = 32*pose_matrix*pose_matrix
activation_in_channels = 32
activation_out_channels = 32

def forward(x):
    batch_size,h,w,c=x.shape
    print(batch_size,h,w,c)