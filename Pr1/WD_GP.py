import torch
import torch.nn as nn
import numpy as np

import sys
import os
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction

import copy


from torch.nn.modules.loss import _Loss



#  TODO: P-1.2 WD with gradient penalty

class distribution_generator(nn.Module):
    def __init__(self,low=0,high=1):
        super(distribution_generator, self).__init__()

        self.low=torch.Tensor([low])
        self.high=torch.Tensor([high])
        self.uniform=torch.distributions.uniform.Uniform(self.low,self.high)

    def WD_lip(self,x,y):
        assert x.size()==y.size()

        a=self.uniform.sample(x.size()).view(x.size())
        z=a*x+(1-a)*y

        return z


class WDGPLoss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, reduction="mean",lambda_=10):
        super(WDGPLoss, self).__init__()
        self.reduction = reduction
        self.lambda_=lambda_

    def forward(self,x,y,d_z):
        loss=-(torch.mean(x,dim=0)-torch.mean(y,dim=0)-self.lambda_*torch.mean((torch.norm(d_z,p=2,dim=-1,keepdim=True)-1)**2,dim=0))

        return loss




