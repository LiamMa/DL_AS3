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
#
#
#
#
# class testloss(_Loss):
#     __constants__ = ['reduction']
#
#     def __init__(self, reduction="mean"):
#         super(testloss, self).__init__()
#         self.reduction = reduction
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     def forward(self,input):
#         loss_=input
#
#         if self.reduction=="mean":
#             loss=torch.mean(loss_)
#         else:
#             loss=torch.sum(loss_)
#
#         return loss

