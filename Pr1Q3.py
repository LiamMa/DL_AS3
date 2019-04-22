from Pr1.Discriminator import MLP as MLP
import Pr1.WD_GP as WD
import Pr1.JSD as JSD

import torch
import sys
import numpy as np
import time
import utils.samplers as samplers
import os
from torch.autograd import grad as torch_grad

path=sys.path[0]


path_save=os.path.join(path,"Prb_1_state")
if not os.path.isdir(path_save):
    os.mkdir(path)




batch_size=512


phi_=np.arange(-1,1.1,0.1)

JSD_list=[]
WD_list=[]

for phi in phi_:

    # TODO: JSD-----------------------------------------------

    print("Define MLP")
    # TODO:Q3
    MLP1= MLP(input_size=2, output_size=1, hidden_size=32, layers=5,activation=True)



    # TODO: ----- inputs p(x); targets q(x)
    print("Define Loss")
    loss=JSD.JSDLoss()



    epochs=100000

    print("Define probability 1 & 2")
    P1=samplers.distribution1(x=0)
    P2=samplers.distribution1(x=phi)


    patience=10000

    # # TODO: ---- training
    print("------- training -------")

    # Check Cuda
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    MLP1=MLP1.to(device)



    optimizer=torch.optim.SGD(params=MLP1.parameters(),lr=1e-3)

    criterion=loss

    # start training

    train_losses_history = []



    print("---- Start Training ----")

    inputs_list =[]
    targets_list=[]
    objective=-np.inf
    p_count=0
    for epoch in range(epochs):

        total = 0
        # ------------ Training ------------
        tic = time.time()
        MLP1.train()

        inputs=torch.Tensor(P1.__next__())
        targets=torch.Tensor(P2.__next__())
        # inputs_list.append(inputs)
        # targets_list.append(targets)


        assert inputs.size()==targets.size()


        if cuda_available:
            inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs1 = MLP1.forward(inputs)
        outputs2 = MLP1.forward(targets)

        loss = criterion(outputs1, outputs2)
        loss.backward()
        optimizer.step()

        loss = loss.item()
        total += targets.size(0)

        if epoch%500==0:
            print('phi: %f---[Epoch %d - Training] Objective_function=%f time: %f' % (phi,epoch, -loss, time.time() - tic))

        if -loss>objective:
            objective=-loss
            p_count=0
            torch.save(MLP1.state_dict(), os.path.join(path_save, "JSD_"+str(phi)))
            state_dict=MLP1.state_dict()
        else:
            p_count+=1
        if p_count>patience:
            break

    # MLP1.load_state_dict(state_dict)

    # TODO:
    #
    # -torch.log(torch.Tensor([2]).to(self.device)) - 1 / 2 * torch.mean(torch.log(input), dim=0)
    print("\n The maximize objective function:  %f"%objective)

    # Compute JSD

    JSD_list.append(objective)













    # TODO: WD_GP-----------------------------------------------



    print("Define MLP")
    # TODO:Q3
    MLP2= MLP(input_size=2, output_size=1, hidden_size=32, layers=5,activation=True)



    # TODO: ----- inputs p(x); targets q(x)
    print("Define Loss")
    loss=WD.WDGPLoss()



    print("Define probability 1 & 2")
    P1=samplers.distribution1(x=0)
    P2=samplers.distribution1(x=phi)
    a_dist=samplers.distribution5()


    # TODO: ---- training
    print("------- training -------")

    # Check Cuda
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MLP2=MLP2.to(device)

    # enable parallel for multi-GPUs


    optimizer=torch.optim.SGD(params=MLP2.parameters(),lr=1e-3)

    criterion=loss

    # start training

    train_losses_history = []



    print("---- Start Training ----")

    inputs_list =[]
    targets_list=[]
    objective=-np.inf
    p_count=0
    for epoch in range(epochs):

        total = 0
        # ------------ Training ------------
        tic = time.time()
        MLP2.train()

        inputs=torch.Tensor(P1.__next__())
        targets=torch.Tensor(P2.__next__())
        a=torch.Tensor(a_dist.__next__())
        a,_=torch.broadcast_tensors(a,inputs)
        z=a*inputs+(1-a)*targets

        assert inputs.size()==targets.size()


        if cuda_available:
            inputs, targets,z = inputs.to(device), targets.to(device),z.to(device)

        optimizer.zero_grad()
        inputs, targets,z = torch.autograd.Variable(inputs), torch.autograd.Variable(targets),torch.autograd.Variable(z,requires_grad=True)
        T_x = MLP2.forward(inputs)
        T_y = MLP2.forward(targets)

        T_z=MLP2.forward(z)

        gradients = torch_grad(outputs=T_z, inputs=z,
                               grad_outputs=torch.ones(
                                   T_z.size()).to(device),
                               create_graph=True, retain_graph=True)[0]



        grad_z=gradients.view(z.size(0),-1)


        loss = criterion(T_x, T_y,grad_z)

        loss.backward()
        optimizer.step()

        loss = loss.item()
        total += targets.size(0)

        if epoch%500==0:
            print('phi: %f---[Epoch %d - Training] Objective_function=%f time: %f' % (phi,epoch, -loss, time.time() - tic))

        if -loss>objective:
            objective=-loss
            p_count=0
            torch.save(MLP2.state_dict(), os.path.join(path_save, "WD_"+str(phi)))
            state_dict=MLP2.state_dict()
        else:
            p_count+=1
        if p_count>patience:
            break



    print("\n The maximize objective function:  %f"%objective)


    # Compute WD


    MLP2.eval()

    assert inputs.size() == targets.size()

    if cuda_available:
        inputs, targets, z = inputs.to(device), targets.to(device), z.to(device)

    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    T_x = MLP2.forward(inputs)
    T_y = MLP2.forward(targets)


    WD_value=torch.mean(T_x,dim=0)-torch.mean(T_y,dim=0)


    WD_list.append(WD_value.item())


print("JSD: ")
print(JSD_list)
print("WD_GP: ")
print(WD_list)

JSD_axis=np.array(JSD_list)
WD_axis=np.array(WD_list)

np.save(os.path.join(path,"JSD_axis.npy"),JSD_axis)
np.save(os.path.join(path,"WD_axis.npy"),WD_axis)

print("Save npy done")