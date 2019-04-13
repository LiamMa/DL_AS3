
import torch
import torch.nn as nn
import copy
import time
import numpy as np
import os


class trainer(object):
    def __init__(self,model,P1,P2,loss,path="",epochs=100,patience_level=5,filename="___"):
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
        self.P1 = P1
        self.P2 = P2
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



            inputs=torch.Tensor(self.P1.__next__())
            targets=torch.Tensor(self.P2.__next__())
            assert inputs.size()==targets.size()


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
            total += targets.size(0)


            print('[Epoch %d - Training] before traine loss=%f time: %f' % (epoch, loss, time.time() - tic),end="\r")



            train_losses_history.append(loss)


            # ------------ Validation ------------
            model.eval()
            total = 0
            valid_loss=0


            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs1 = model.forward(inputs)
            outputs2 = model.forward(targets)

            total += targets.size(0)
            valid_loss=criterion(outputs1,outputs2).item()*targets.size(0)

            valid_loss=valid_loss
            valid_loss_history.append(valid_loss)


            print('[Epoch %d -- Trained] Valid__loss=%f  time: %f' %
                  (epoch, valid_loss, time.time() - tic))




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




