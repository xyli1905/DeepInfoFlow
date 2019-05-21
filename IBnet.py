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
from json_parser import JsonParser
from logger import *
import utils
import datetime

import time


class SaveActivations:
    def __init__(self):
        self._opt = BaseOption().parse()
        # check device
        self._device = torch.device("cpu" if torch.cuda.is_available() else "cpu") # device setup
        print("device: ",self._device)


    def apply_opt(self):
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
            raise RuntimeError('Do not have {name} dataset, Please be sure to use the existing dataset'.format(name = self._opt.dataset))

        # construct saving directory
        save_root_dir = self._opt.save_root_dir
        dataset = self._opt.dataset
        time = datetime.datetime.today().strftime('%m_%d_%H_%M')
        model = ''.join(list(map(lambda x:str(x) + '_', self._model.layer_dims)))
        folder_name = dataset + '_'+self._opt.experiment_name + '_Time_' + time + '_Model_' + model
        self._path_to_dir = save_root_dir + '/' + folder_name + '/'
        print(self._path_to_dir)
        if not os.path.exists(self._path_to_dir):
            os.makedirs(self._path_to_dir)

        self._logger = Logger(opt=self._opt, plot_name = folder_name)
        self._json = JsonParser()

    def _update_opt(self, other):
        for key, val in other.items():
            setattr(self._opt, key, val)


    def _initialize_model(self, dims):
        # weight initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        # model construction
        self._model = Model(activation = self._opt.activation, dims = dims, train=True)
        self._model.apply(weights_init)
        # optimizer
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._opt.lr)
        # loss
        self._criterion = nn.CrossEntropyLoss() # loss


    def training_model(self):

        self.apply_opt()
        self._json.dump_json(opt=self._opt, path=self._path_to_dir)
        print('Begin training...')
        self._model = self._model.to(self._device)

        save_step = 100

        eta = 1.
        running_loss = 0.0
        running_acc = 0.0
        t_begin = time.time()

        # main loop for training
        for i_epoch in range(self._opt.max_epoch):
            # set to train
            self._model.train()

            # train batch
            if ((i_epoch+1) % save_step == 0) or (i_epoch == 0):
                print('\n{}'.format(11*'------'))
            
            running_acc = 0
            running_acc = 0
            for i_batch , (inputs, labels) in enumerate(self._train_set):
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                bsize = inputs.shape[0]
                # set to learnable
                with torch.set_grad_enabled(True):
                    #forward
                    self._optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    corrects = torch.sum(preds == labels.data).double()

                    # backprop
                    loss.backward()
                    self._optimizer.step()

                self._logger.log(self._model)# each time update weights LOG IT!

                # monitor the running loss & running accuracy
                # eta = eta / (1. + bsize*eta)
                # running_loss = (1. - bsize*eta)*running_loss + eta*loss.detach()
                # running_acc = (1. - bsize*eta)*running_acc + eta*corrects.detach()
                running_acc = float(corrects.detach() / bsize)
                running_loss = float(loss.detach())
                if ((i_epoch+1) % save_step == 0) or (i_epoch == 0):
                    output_format = "\repoch:{epoch} batch:{batch:2d} " +\
                                    "Loss:{loss:.5e} Acc:{acc:.5f}% " +\
                                    "numacc:{num:.0f}/{tnum:.0f}"
                    print(output_format.format(batch=i_batch+1,
                                               epoch=i_epoch+1,
                                               loss=running_loss,
                                               acc=running_acc*100.,
                                               num=corrects,
                                               tnum=bsize))
            
            self._model.eval()
            for i_batch , (inputs, labels) in enumerate(self._test_set):
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                bsize = inputs.shape[0]
                    #forward
                outputs = self._model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == labels.data).double()

                # monitor the running loss & running accuracy
                # eta = eta / (1. + bsize*eta)
                # running_loss = (1. - bsize*eta)*running_loss + eta*loss.detach()
                # running_acc = (1. - bsize*eta)*running_acc + eta*corrects.detach()
                running_acc = float(corrects.detach() / bsize)
                if ((i_epoch+1) % save_step == 0) or (i_epoch == 0):
                    output_format = "\repoch:{epoch} batch:{batch:2d} " +\
                                    "Acc:{acc:.5f}% " +\
                                    "numacc:{num:.0f}/{tnum:.0f}"
                    print(output_format.format(batch=i_batch+1,
                                               epoch=i_epoch+1,
                                               acc=running_acc*100.,
                                               num=corrects,
                                               tnum=bsize))

            self._logger.update(i_epoch)# to calculate std and mean

            if ((i_epoch+1) % save_step == 0) or (i_epoch == 0):
                print('{}'.format(11*'------'))
                t_end = time.time()
                print('time cost for this output period: {:.3f}(s)'.format(t_end - t_begin))
                t_begin = time.time()

            # saving model for each epoch
            self.save_model(i_epoch)

            # print(self._logger)
        self._logger.plot_mean_std()
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
