from Pr1.Discriminator import MLP as MLP
import Pr1.WD_GP as WD
import Pr1.JSD as JSD

import torch
import torch.nn as nn
from utils.trainer import trainer
import sys

import utils.samplers as samplers


path=sys.path[0]

batch_size=512


print("Define MLP")
# TODO: Q1
MLP1= MLP(input_size=1, output_size=1, hidden_size=32, layers=3)



# TODO: ----- inputs p(x); targets q(x)
print("Define Loss")
loss=JSD.JSDLoss()


print('Set trainer')
trainer1=trainer(model=MLP1,P1=samplers.distribution3(),P2=samplers.distribution4(batch_size=512),epochs=10,loss=loss,path=path,filename="JSD",)

print("----- start training -----")
loss,state_dict=trainer1.GAN_train_()
print("----- finish training -----")




MLP1.load_state_dict(state_dict,False)



