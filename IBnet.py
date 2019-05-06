from __future__ import print_function
import numpy as np
import sys
import os, pickle
from collections import defaultdict, OrderedDict

import simplebinmi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')

import torch
import torch.nn as nn
import torch.optim as optim

from model import Model
from base_options import BaseOption
from logger import *
import utils
import datetime


class SaveActivations:
    def __init__(self):
        self._opt = BaseOption().parse()
        # check device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
        print("device: ",self._device)


        # dataset
        if self._opt.dataset == "MNIST":
            train_data, test_data = utils.get_mnist()
            self._train_set = torch.utils.data.DataLoader(train_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            self._test_set = torch.utils.data.DataLoader(test_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            self._initialize_model(dims = self._opt.layer_dims)
            print("MNIST experiment")

        elif self._opt.dataset == "IBNet":
            train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
            test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
            self._train_set = torch.utils.data.DataLoader(train_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            self._test_set = torch.utils.data.DataLoader(test_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            self._initialize_model(dims = self._opt.layer_dims)
            print("IBnet experiment")
        else:
            raise RuntimeError('Do not have {name} dataset, Please be sure to use the existing dataset'.format(name = dataset))

        # construct saving directory
        save_root_dir = self._opt.save_root_dir
        dataset = self._opt.dataset
        time = datetime.datetime.today().strftime('%m_%d_%H_%M')
        model = ''.join(list(map(lambda x:str(x) + '_', self._model.layer_dims)))
        self._path_to_dir = save_root_dir + '/' + dataset + '_Time_' + time + '_Model_' + model + '/'
        if not os.path.exists(self._path_to_dir):
            os.makedirs(self._path_to_dir)
        
        

    def _update_opt(self, other):
        for key, val in other.items():
            setattr(self._opt, key, val)
        self._logger = Logger(opt = self._opt)

    def _initialize_model(self, dims):
        # weight initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        # model construction
        self._model = Model(dims = dims, train=True)
        self._model.apply(weights_init)
        # optimizer 
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._opt.lr, momentum=self._opt.momentum)
        # loss
        self._criterion = nn.CrossEntropyLoss() # loss


    def training_model(self):
        print('Begin training...')
        self._model.to(self._device)

        # main loop for training
        for i in range(0, self._opt.max_epoch):
            # set to train
            self._model.train()
            running_loss = 0.0
            running_acc = 0.0
            # batch training
            for j , (inputs, labels) in enumerate(self._train_set):
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                # set to learnable
                with torch.set_grad_enabled(True):
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    self._optimizer.zero_grad()

                    loss.backward()

                    self._optimizer.step()

                    # logging for std mean and L2N
                    self._logger.log(self._model)

                    #break # for debug purpose


                # acc and loss calculation
                running_loss += loss * inputs.size(0)
                corrects = torch.sum(preds == labels.data).double()
                running_acc += corrects
                sys.stdout.flush()
                print('\repoch:{epoch} Loss: {loss:.6f} acc:{acc:.6f}'.format(epoch=i+1, loss=loss, acc=corrects), end="")
            
            self._logger.update(i)
            
            epoch_loss = running_loss / len(self._train_set)
            epoch_acc = running_acc.double() / len(self._train_set)
            print("")
            print('------------------summary epoch {epoch} ------------------------'.format(epoch = i+1))
            print('Loss {loss:.6f} acc:{acc:.6f}'.format( loss=epoch_loss, acc=epoch_acc))
            print('----------------------------------------------------------------')
            # saving model
            # uncomment to save model
            self.save_model(i)

            # print(self._logger)
        print ('-------------------------training end--------------------------')

    def save_model(self, epoch):
        save_full_path = self.generate_save_fullpath(epoch + 1)
        torch.save({
        'epoch': epoch,
        'model_state_dict': self._model.state_dict(),
        'optimizer_state_dict': self._optimizer.state_dict(),
        }, save_full_path)


    
    def generate_save_fullpath(self, epoch):
        suffix = '.pth'
        fullpath = self._path_to_dir + 'model_epoch_' + str(epoch) + suffix
        return fullpath

if __name__ == "__main__":
    t = SaveActivations()
    t._update_opt({})
    t.training_model()
    
    # loss = nn.CrossEntropyLoss()
    # inputs = torch.randn(64, 2, requires_grad=True)
    # target = torch.empty(64, dtype=torch.long).random_(2)
    # output = loss(inputs, target)
    # print(inputs)
    # print(target)
    # print(output)
