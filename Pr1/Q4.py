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

class Q4Loss(_Loss):
    __constants__ = ['reduction']
    def __init__(self,reduction="mean"):
        super(Q4Loss, self).__init__()
        self.reduction=reduction

    def forward(self,f0_out,f1_out):
        loss=-(torch.mean(torch.log(f1_out),dim=0)+torch.mean(torch.log(1-f0_out),dim=0))
        return loss
#
#
#
#

