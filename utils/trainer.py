
import torch
import torch.nn as nn
import copy
import time
import numpy as np
import os


class trainer(object):
    def __init__(self,model,train_loader,loss,valid_loader=None,path="",epochs=100,patience_level=5,filename="___"):
        '''
        The trainer for ANT.
        Growth stage: train_growth()
        Refinement stage: refine()

        :param model:  The model
        :param train_loader:  the training loader
        :param valid_loader:  the validation loader
        :param path:  the save path
        :param epochs: the training epochs
        '''
        super(trainer,self).__init__()
        self.model = model
        self.node_dict = model.node_dict
        self.node_list = model.node_list
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.path = path
        self.epochs = epochs
        self.patience_level = patience_level
        self.loss=loss
        self.filename=filename

    def GAN_train_(self):

        '''
        To train the copy of the model with/withuot adding modules.
            To be used in growth stage to compare different module.
            To be used in refinement stage to train whole mdoel
        Only be used internally
        :param node_name: the name of the node to train; If == "", train the whole model
        :return: Return validation_acc,
                parameter/state_dict(if node_name specified, return node.state_dict();
                else return model.state_dict()
                !!
                The state_dict is in cpu
        '''
        filename=self.filename
        # initialize path
        path = self.path


        # copy a version of model to avoid collapse
        model = copy.deepcopy(self.model)
        #

        # Check Cuda
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # enable parallel for multi-GPUs
        parallel=False
        if cuda_available:
            if cuda_count > 1:
                print("Let's use", cuda_count, "GPUs!")
                model = nn.DataParallel(model)
                parallel=True
        model.to(device)

        optimizer=torch.optim.SGD(params=model.parameters(),lr=1e-3)




        patience=self.patience_level

        train_loader=self.train_loader

        valid_loader=self.valid_loader

        if self.valid_loader==None:
            valid=False

        criterion=self.loss



        # start training

        valid_loss_history=[]
        train_losses_history = []
        patience_count = 0
        valid_loss_last=np.inf

        print("---- Start Training ----")

        for epoch in range(self.epochs):

            losses = []
            total = 0
            # ------------ Training ------------
            tic = time.time()
            model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if targets.size().__len__() > 1:
                    if targets.size(1) == 1:
                        targets = targets.flatten().type(torch.long)
                if cuda_available:
                    inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs1 = model.forward(inputs)
                outputs2 = model.forward(targets)

                loss = criterion(outputs1, outputs2)
                loss.backward()
                optimizer.step()

                loss = loss.item()
                losses.append(loss*inputs.size(0))
                total += targets.size(0)

                if batch_idx % 100 == 0:
                    curr_loss = np.mean(losses)
                    print('[Epoch %d - batch %d] loss=%f time: %f' % (epoch, batch_idx, curr_loss, time.time() - tic),end="\r")


            train_loss = np.sum(losses)/total

            train_losses_history.append(train_loss)


            # ------------ Validation ------------
            model.eval()
            if valid:
                total = 0
                valid_loss=0
                for batch_idx, (inputs, targets) in enumerate(valid_loader):
                    if targets.size().__len__()>1:
                        if targets.size(1)==1:
                            targets = targets.flatten().type(torch.long)

                    if cuda_available == True:
                        inputs, targets = inputs.to(device), targets.to(device)

                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                    outputs1 = model.forward(inputs)
                    outputs2 = model.forward(targets)

                    total += targets.size(0)
                    valid_loss+=criterion(outputs1,outputs2).item()*targets.size(0)

                valid_loss=valid_loss/total
                valid_loss_history.append(valid_loss)


                print('[Epoch %d] train_loss=%f  val_loss=%f  time: %f ' %
                      (epoch, train_loss, valid_loss, time.time() - tic))

            else:
                total = 0
                valid_loss = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if targets.size().__len__() > 1:
                        if targets.size(1) == 1:
                            targets = targets.flatten().type(torch.long)

                    if cuda_available == True:
                        inputs, targets = inputs.to(device), targets.to(device)

                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                    outputs1 = model.forward(inputs)
                    outputs2 = model.forward(targets)

                    total += targets.size(0)
                    valid_loss += criterion(outputs1, outputs2).item() * targets.size(0)


                valid_loss = valid_loss / total
                valid_loss_history.append(valid_loss)

                print('[Epoch %d] train_loss=%f  real_train_loss=%f  time: %f ' %
                      (epoch, train_loss, valid_loss, time.time() - tic))

            # TODO: Patience Module



            # TODO: Validation Loss base and save best valid loss model
            if valid_loss < valid_loss_last:
                if cuda_count > 1:
                    torch.save(model.module.state_dict(), os.path.join(path, filename+"paral"))
                else:
                    torch.save(model.state_dict(), os.path.join(path, filename))
                patience_count = 0
                valid_loss_last = valid_loss
            else:
                patience_count += 1

            if patience_count >= patience:
                print("\nEarly Stopping the Training ---- Stopping Criterion:  Loss" )
                break
            # #################################




        # reload model to get new state_dict
        # TODO: --- enable early stopping for refinement (ONLY SAVE BEST BUT NOT STOP)
        # if not refine: # TODO: add this line to disable early stopping for refinement
        if cuda_count > 1:
            model.module.load_state_dict(torch.load(os.path.join(path, filename+"paral")))
        else:
            model.load_state_dict(torch.load(os.path.join(path,filename)))



        valid_loss=valid_loss_last
        print("----- Best Valid/Train_Loss: %f \n   "%(valid_loss_last)) # for loss criterion but save best acc

        state_dict=model.state_dict()
        return valid_loss, state_dict




