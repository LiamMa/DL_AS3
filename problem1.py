from Pr1.Discriminator import MLP as MLP
import Pr1.WD_GP as WD
import Pr1.JSD as JSD

import torch
import torch.nn as nn
from utils.trainer import trainer
import sys

path=sys.path[0]

batch_size=512


# TODO: Q1
MLP1= MLP(input_size=1, output_size=1, hidden_size=32, layers=3)

optimizer1=torch.optim.SGD(params=MLP1.parameters(),lr=1e-3)


train_loader=""
# TODO: ----- inputs p(x); targets q(x)

loss=JSD.JSDLoss()

trainer1=trainer(model=MLP1,train_loader=train_loader,loss=loss,path=path,filename="JSD")

loss,state_dict=trainer.train_()

MLP1.load_state_dict(state_dict,False)



