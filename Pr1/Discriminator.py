import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
import copy
from torch.nn.modules.loss import _Loss

class MLP(nn.Module):
    def __init__(self,input_size,output_size,layers=3,hidden_size=None,
                 to_softmax=True,activation=False,dp=0):
        '''
        :param input_size:  Input size
        :param output_size:  Output size
        :param layers:  The number of layers
        :param hidden_size:  A list of ints; The sizes of hidden layers. If None, the same as the inputs size.
                            If an int, all hidden layers has same size as the int.
        '''
        super(MLP, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.layers=layers
        self.to_softmax=to_softmax
        self.add_activation=activation
        self.dp=dp
        if hidden_size==None:
            hidden_size=input_size
        if type(hidden_size)==list:
            assert hidden_size.__len__()==self.layers
            self.hidden_size=hidden_size
        else:
            assert hidden_size>0
            self.hidden_size=[int(hidden_size) for _ in range(layers)]
        sizes=[input_size]+self.hidden_size+[output_size]

        self.MLP=nn.Sequential()
        for i in range(self.layers+1):
            self.MLP.add_module("layer_"+str(i),nn.Linear(sizes[i],sizes[i+1]))
            if self.add_activation and i<self.layers:
                # self.MLP.add_module("sigmoid"+str(i),nn.Sigmoid())
                self.MLP.add_module("ReLU"+str(i),nn.ReLU())
                # self.MLP.add_module("leakyReLU" + str(i),nn.LeakyReLU(negative_slope=0.01))
                # self.MLP.add_module("tanh"+str(i),nn.Tanh())
            if self.dp>0:
                self.MLP.add_module("Dropout"+str(i),nn.Dropout(self.dp))
        if self.to_softmax:
            if output_size==1:
                self.MLP.add_module("sigmoid", nn.Sigmoid())
            else:
                self.MLP.add_module("softmax",nn.Softmax(dim=-1))

                
    def forward(self, input):
        return self.MLP(input)



