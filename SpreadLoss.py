import torch
from torch.nn.modules.loss import _Loss


class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, number_of_output_classes=10):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.number_of_output_classes = number_of_output_classes

    def forward(self, x, target, r):
        b, E = x.shape
        assert E == self.number_of_output_classes
        margin = self.m_min + (self.m_max - self.m_min)*r

        if type(x) == torch.cuda.FloatTensor:
            at = torch.cuda.FloatTensor(b).fill_(0)
        else:
            at = torch.FloatTensor(b).fill_(0)

        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        return loss