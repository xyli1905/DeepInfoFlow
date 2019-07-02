
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import DenseNet
from logger import *
from basemodel import BaseModel



class SeqModel(BaseModel):
    def __init__(self, IS_TRAIN = True, model_path = None):
        super(SeqModel, self).__init__(IS_TRAIN, model_path)
        self._name = "seq model"

    def eval(self):
        self._network.eval()


    def set_network(self, network):
        '''
        Densnet 
        '''
        self._network = DenseNet(self._opt, train=self._is_train)

    def set_optimizer(self):
        '''
        SGD ADAM
        '''
        if self._opt.optimizer == 'sgd':
            self._optimizer = optim.SGD(self._network.parameters(), lr=self._opt.lr, momentum=self._opt.momentum)
        elif self._opt.optimizer == 'adam':
            self._optimizer = optim.Adam(self._network.parameters(), lr=self._opt.lr)

    def set_lossfunction(self):
        '''
        Crossentrop or self-defined classes
        '''
        if self._opt.lossfunc == 'crossentropy':
            self._criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('presently only support CrossEntropy')
    
    
    def train_epoch(self, i_epoch, probe):
        '''
        standard one
        '''
        # set to train
        self._network.train()

        for i_batch , (inputs, labels) in enumerate(self._train_set):
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            bsize = inputs.shape[0]
            # set to learnable
            with torch.set_grad_enabled(True):
                #forward
                self._optimizer.zero_grad()
                outputs = self._network(inputs)
                loss = self._criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == labels.data).double()

                # backprop
                loss.backward()
                self._optimizer.step()

            if self.need_log(i_epoch):
                self._logger.log(self._network)# each time update weights LOG IT!

            # monitor the accumulated accuracy
            probe.update_acc(acc=float(corrects.detach()))
            probe.update_loss(loss=float(bsize*loss.detach()))
    

    def test_epoch(self, i_epoch, probe):
        # set to test
        self._network.eval()

        for i_batch , (inputs, labels) in enumerate(self._test_set):
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            bsize = inputs.shape[0]
            #forward
            outputs = self._network(inputs)
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data).double()

            # monitor the accumulated accuracy
            probe.update_acc(acc=float(corrects.detach()))


    def predict(self, batch_input):
        if self._is_train:
            raise ValueError('predict only applie for _is_train=False')

        return self._network(batch_input)


    def save_model(self, i_epoch):
        self._save_model(self._network, self._optimizer, i_epoch)

    def load_model(self, epoch_file):
        self._load_model(self._network, self._optimizer, epoch_file)




if __name__ == "__main__":
    t = IBmodel()
    t._update_opt({})
    t.training_model()
    
    # loss = nn.CrossEntropyLoss()
    # inputs = torch.randn(64, 2, requires_grad=True)
    # target = torch.empty(64, dtype=torch.long).random_(2)
    # output = loss(inputs, target)
    # print(inputs)
    # print(target)
    # print(output)
