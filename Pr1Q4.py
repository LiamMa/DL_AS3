from Pr1.Discriminator import MLP as MLP
import Pr1.Q4 as Q4

import torch
import sys
import numpy as np
import time
import utils.samplers as samplers
import os

import numpy as np

import torch
import matplotlib.pyplot as plt


print("Load modules done")

path=sys.path[0]

path_save=os.path.join(path,"Prb_1_Q4")
#
if not os.path.exists(path_save):
    os.mkdir(path_save)

batch_size=512



# TODO: JSD-----------------------------------------------

print("Define MLP")
# TODO:Q3
# MLP4= MLP(input_size=1, output_size=1, hidden_size=32, layers=4)
MLP4= MLP(input_size=1, output_size=1, hidden_size=128, layers=6,activation=True)

print(MLP4)

# TODO: ----- inputs p(x); targets q(x)
print("Define Loss")
criterion=Q4.Q4Loss()


# epochs=99990000
epochs=300000


print("Define probability 1 & 0")
P0=samplers.distribution3(batch_size=512)
P1=samplers.distribution4(batch_size=512)



patience=1000

# TODO: ---- training
print("------- training -------")

# Check Cuda
cuda_available = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MLP4=MLP4.to(device)



optimizer=torch.optim.SGD(params=MLP4.parameters(),lr=1e-3)


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
    MLP4.train()

    inputs=torch.Tensor(P0.__next__())
    targets=torch.Tensor(P1.__next__())



    assert inputs.size()==targets.size()


    if cuda_available:
        inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    outputs0 = MLP4.forward(inputs)
    outputs1 = MLP4.forward(targets)

    # print(outputs0)
    # print(outputs1)

    loss = criterion(outputs0, outputs1)
    loss.backward()
    optimizer.step()

    loss = loss.item()
    total += targets.size(0)

    if epoch%500==0:
        print('[Epoch %d - Training] Objective_function=%f time: %f' % (epoch, -loss, time.time() - tic))

    if -loss>objective:
        print('[Epoch %d - Training] Objective_function=%f time: %f' % (epoch, -loss, time.time() - tic))
        objective=-loss
        p_count=0
        torch.save(MLP4.state_dict(), os.path.join(path_save, "Q4_discriminator_2000"))
        state=MLP4.state_dict()

    else:
        p_count+=1
    if p_count>patience:
        print("Early stopping at %d epochs"%epoch)
        break

print("\n The maximize objective function:  %f"%objective)




MLP4.load_state_dict(state)
#






# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x * 2 + 1) + x * 0.75
d = lambda x: (1 - torch.tanh(x * 2 + 1) ** 2) * 2 + 0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5, 5)

# exact
xx = np.linspace(-5, 5, 1000)
N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
plt.plot(xx, N(xx))
# plt.title("The f0 and f1")
plt.legend(['f1', 'f0'])
plt.savefig("p0_p1_.jpg")


############### import the sampler ``samplers.distribution4''
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######


# the discriminater

# TODO: Load Discirminater

############### plotting things
############### (1) plot the output of your trained discriminator
############### (2) plot the estimated density contrasted with the true density

tensor_xx=torch.Tensor(xx).to(device).view(-1,1)
r = MLP4(tensor_xx).detach().cpu().numpy().reshape(-1) # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(xx, r)
plt.title(r'$D(x)$')

# estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
# replace "np.ones_like(xx)*0." with your estimate


estimated =N(xx)*r/(1-r)
# TODO: load f_0 density function

plt.subplot(1, 2, 2)
plt.plot(xx, estimated)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
plt.legend(['Estimated', 'True'])
plt.title('Estimated vs True')
plt.savefig("Q4plot_relu.jpg")
# plt.savefig("Q4plot_tanh.jpg")
# plt.savefig("Q4plot_relu_nostop.jpg")


print("plot done")



plt.figure(figsize=(8, 4))
plt.plot(xx, estimated)
plt.plot(xx,N(xx))
plt.legend(['Estimated', 'f0'])
plt.savefig("estimated_f0_relu.jpg")
# plt.savefig("estimated_f0_tanh.jpg")
# plt.savefig("estimated_f0_relu_nostop.jpg")


