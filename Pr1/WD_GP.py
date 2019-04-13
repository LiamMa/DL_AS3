import torch
import torch.nn as nn
import numpy as np

import sys
import os
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction

import copy


from torch.nn.modules.loss import _Loss




# TODO: P-1.1 JSD

class JSDLoss(_Loss):
    __constants__ = ['reduction']
    def __init__(self,reduction="mean"):
        super(JSDLoss, self).__init__()
        self.reduction=reduction
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self,input,target):
        loss_=-torch.log(torch.Tensor([2]))-1/2*torch.mean(torch.log(input),dim=-1)-1/2*torch.mean(torch.log(1-target),dim=-1)

        if self.reduction=="mean":
            loss=torch.mean(loss_)
        else:
            loss=torch.sum(loss_)

        return loss


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lambda_=lambda_

    def forward(self,x,y,d_z):
        loss_=-torch.mean(x,dim=-1)+torch.mean(y,dim=-1)-self.lambda_*torch.dist(d_z,torch.Tensor([1]),p=2)

        if self.reduction=="mean":
            loss=torch.mean(loss_)
        else:
            loss=torch.sum(loss_)

        return loss




def gradient_T(model):
    model=copy.deepcopy(model)
    for i,j in model.named_parameters():
        print(i)
        print(j)

    input_size=model.input_size
    x=torch.ones([1,input_size])
    x=x.requires_grad_(True)
    y=model(x)
    y.backward(torch.ones(y.size()))
    print(y)
    print(x)
    print(x.grad)
    return x.grad



