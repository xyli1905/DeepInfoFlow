import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from base_options import BaseOption
from logger import *
import utils
from dataLoader import DataProvider

import datetime
import time
import os

import json
import argparse


class BaseModel:
    def __init__(self, IS_TRAIN = True, model_path = None):
        self._name = "base model"
        self._device = torch.device("cpu" if torch.cuda.is_available() else "cpu") # device setup
        print("device: ",self._device)

        self._is_train = IS_TRAIN
        self._model_saved = False
        self._model_loaded = False
        self._json_saved = False
        self._json_loaded = False

        self._save_step = 100

        self._get_option(model_path) # prepare self._opt

        self._build_model() #set network optimizer lossfunction
        self._set_model_directory(model_path)
    

    def get_opt(self):
        return self._opt

    def eval(self):
        raise NotImplementedError('eval should be specified in individual model')

    def _initialize_model(self):
        raise NotImplementedError('presently use default pytorch initalization')

    def _build_model(self):
        self.set_network()
        self.set_optimizer()
        self.set_lossfunction()

    def set_network(self):
        raise NotImplementedError('set_network should be specified in individual model')

    def set_optimizer(self):
        raise NotImplementedError('set_optimizer should be specified in individual model')

    def set_lossfunction(self):
        raise NotImplementedError('set_lossfunction should be specified in individual model')


    def _load_dataset(self):
        '''
        using self-defined classes
        '''
        self._name = f"{self._name}_{self._opt.dataset}"

        dp = DataProvider(dataset_name = self._opt.dataset, 
                          batch_size = self._opt.batchsize, 
                          num_workers = self._opt.num_workers, 
                          shuffle = True)
        self._train_set, self._test_set = dp.get_train_test_data()

        self._train_size = len(self._train_set)
        self._test_size = len(self._test_set)


    def train_model(self):
        if not self._is_train:
            raise ValueError('train_model only applie for _is_train=True')

        self._load_dataset()
        self._save_opt_to_json()
        
        probe = Monitor(self._train_size, self._test_size, save_step = self._save_step)
        self._logger = Logger(opt = self._opt, plot_name = self._model_name)

        print('Begin training...')
        for i_epoch in range(self._opt.max_epoch):

            if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                print('\n{}'.format(11*'------'))

            # train one epoch
            probe.initialize()

            self.train_epoch(i_epoch, probe)

            probe.monitor_epoch(i_epoch, mode='train')
            self._logger.log_acc_loss(i_epoch, 'train', acc=probe.epoch_acc, loss=probe.epoch_loss)
            
            # test one epoch
            probe.initialize()

            self.test_epoch(i_epoch, probe)

            probe.monitor_epoch(i_epoch, mode='test')
            self._logger.log_acc_loss(i_epoch, 'test', acc=probe.epoch_acc)

            # update log for selected epoches
            if self.need_log(i_epoch):
                self._logger.update(i_epoch)# to calculate std and mean, svd

            if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                print('{}'.format(11*'------'))
                t_end = time.time()
                print('time cost for this output period: {:.3f}(s)'.format(t_end - t_begin))
                t_begin = time.time()

            # saving model for each epoch
            self.save_model(i_epoch)

        self._logger.plot_figures()
        print('-------------------------training end--------------------------')
    
    def train_epoch(self, i_epoch, probe):
        raise NotImplementedError('train_epoch should be specified in individual model')

    def test_epoch(self, i_epoch, probe):
        raise NotImplementedError('test_epoch should be specified in individual model')

    def predict(self, batch_input):
        raise NotImplementedError('predict should be specified in individual model')


    def need_log(self, epoch)->bool:
        for idx, val in enumerate(self._opt.log_seperator):
            if epoch < val:
                return epoch % self._opt.log_frequency[idx] == 0


    def _set_model_directory(self, mpath):
        if mpath == None:
            # construct saving directory
            save_root_dir = self._opt.save_root_dir
            dataset = self._opt.dataset
            time = datetime.datetime.today().strftime('%m_%d_%H_%M')
            self._model_name = f"{dataset}_{self._opt.experiment_name}_{self._opt.activation}_Time_{time}"

            self._path_to_dir = os.path.join(save_root_dir, self._model_name)
            if not os.path.exists(self._path_to_dir):
                os.makedirs(self._path_to_dir)

            self._model_path = os.path.join(self._path_to_dir, "models")
            if not os.path.exists(self._model_path):
                os.makedirs(self._model_path)
        else:
            # set existed model path
            self._path_to_dir = mpath
            self._model_path = os.path.join(self._path_to_dir, "models")
        
        print(self._path_to_dir)


    def save_model(self, i_epoch):
        '''NOTE interface for individual model instance
        '''
        raise NotImplementedError('save_model should be specified in individual model')

    def _save_model(self, network, optimizer, epoch):
        model_name = f"model_epoch_{str(epoch+1)}.pth"
        save_full_path = os.path.join(self._model_path, model_name)

        torch.save({'epoch': epoch,
                    'network_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_full_path)

        self._model_saved = True


    def load_model(self, epoch_file):
        '''NOTE interface for individual model instance
        '''
        raise NotImplementedError('load_model should be specified in individual model')

    def _load_model(self, network, optimizer, epoch_file, CKECK_LOG):
        ckpt = torch.load(os.path.join(self._model_path, epoch_file))
        epoch = ckpt['epoch']
        load_indicator = {'NEED_LOG': True, 'epoch': epoch}

        if (CKECK_LOG) and (not self.need_log(epoch)):
            load_indicator['NEED_LOG'] = False
            return load_indicator

        # load network epoch weight
        network.load_state_dict(ckpt['network_state_dict'])

        # load optimizer, if re-train
        if self._is_train:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self._model_loaded = True
        return load_indicator


    def _get_option(self, mpath):
        if mpath == None:
            self._opt = BaseOption().parse()
        else:
            self._opt = self._load_json_as_argparse(mpath)


    def _save_opt_to_json(self):
        json_dir = os.path.join(self._path_to_dir, "opt.json")
        argparse_dict = vars(self._opt)
        with open(json_dir, 'w') as outfile:
            json.dump(argparse_dict, outfile)
        print ("configs have been dumped into %s" % json_dir)
        self._json_saved = True

    def _load_json_as_argparse(self, mpath):
        try:
            json_dir = os.path.join(mpath, "opt.json")
            js = open(json_dir).read()
            data = json.loads(js)
            opt = argparse.Namespace()
            for key, val in data.items():
                setattr(opt, key, val) 
            return opt
        except Exception as e:
            print("No such file or directory %s" % (json_dir))
        self._json_loaded = True

    def _update_opt(self, other):
        for key, val in other.items():
            setattr(self._opt, key, val)



class Monitor:
    def __init__(self, train_size, test_size, save_step):
        self._train_size = train_size
        self._test_size = test_size
        self._save_step = save_step

        self.format_train = "\repoch:{epoch} Loss:{loss:.5e} Acc:{acc:.5f}% " +\
                            "numacc:{num:.0f}/{tnum:.0f}"
        self.format_test  = "\repoch:{epoch} Acc:{acc:.5f}% " +\
                            "numacc:{num:.0f}/{tnum:.0f}"

    def initialize(self):
        self.epoch_acc = 0.
        self.epoch_loss = 0.
    
    def update_acc(self, acc):
        self.epoch_acc += acc
        
    def update_loss(self, loss=0.):
        self.epoch_loss += loss
    
    def monitor_epoch(self, i_epoch, mode='train'):
        if mode == 'train':
            avg_acc  = self.epoch_acc / float(self._train_size)
            avg_loss = self.epoch_loss / float(self._train_size)
            if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                print(self.format_train.format(epoch=i_epoch+1,
                                               loss=avg_loss,
                                               acc=avg_acc*100.,
                                               num=self.epoch_acc,
                                               tnum=self._train_size))
        elif mode == 'test':
            avg_acc = self.epoch_acc / float(self._test_size)
            if ((i_epoch+1) % self._save_step == 0) or (i_epoch == 0):
                print(self.format_test.format(epoch=i_epoch+1,
                                              acc=avg_acc*100.,
                                              num=self.epoch_acc,
                                              tnum=self._test_size))



if __name__ == "__main__":
    pass