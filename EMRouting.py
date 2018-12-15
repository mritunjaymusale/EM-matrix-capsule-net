from torch import nn
import torch
import math


class EMRouting():
    def __init__(self, cuda = True):

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.ln_2pi = math.log(2*math.pi)
        self.cuda =cuda

    def m_step(self, a_in, r, v, eps, b, B, C, psize, beta_a, beta_u, _lambda):
        
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(_lambda*(beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
            - torch.log(sigma_sq.sqrt()) \
            - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps, beta_a, beta_u, _lambda, iters):
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        if self.cuda:
            r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        else:
            r = torch.FloatTensor(b, B, C).fill_(1./C)

        for iter_ in range(iters):
            a_out, mu, sigma_sq = self.m_step(
                a_in, r, v, eps, b, B, C, psize, beta_a, beta_u, _lambda)
            if iter_ < iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out
