import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from EMRouting import EMRouting

class ConvCaps(nn.Module):
    

    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3,cuda = True):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters

        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        self.cuda = cuda
        self.EM = EMRouting(cuda=self.cuda)

    

    def crude_convolution2d(self, x, B, K, psize, stride):
        
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx)
                 for k_idx in range(0, K)]
                for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P):
        
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def forward(self, x):
        b, h, w, c = x.shape

        # add patches
        x, oh, ow = self.crude_convolution2d(
            x, self.B, self.K, self.psize, self.stride)
        # transform view
        p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()

        a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()

        p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
        a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
        v = self.transform_view(p_in, self.weights, self.C, self.P)

        # em_routing
        p_out, a_out = self.EM.caps_em_routing(
            v, a_in, self.C, self.eps, self.beta_a, self.beta_u, self._lambda,self.iters)
        p_out = p_out.view(b, oh, ow, self.C*self.psize)
        a_out = a_out.view(b, oh, ow, self.C)
        out = torch.cat([p_out, a_out], dim=3)
        return out
