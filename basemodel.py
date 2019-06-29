
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from base_options import BaseOption
from json_parser import JsonParser
from logger import *
import utils

import datetime
import time

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

        self._get_option(model_path) # prepare self._opt

        self._log_list = []
        self._cal_log_list()

        self._build_model() #set network and/or optimizer lossfunction
        self._set_model_directory()


    def _initialize_model(self):
        raise NotImplementedError('presently use default pytorch initalization')

    def _build_model(self):
        self._set_network()
        self._set_optimizer()
        self._set_lossfunction()

    def _set_network(self):
        raise NotImplementedError('should be specified in individual model')

    def _set_optimizer(self):
        '''
        SGD
        ADAM
        '''
        pass

    def _set_lossfunction(self):
        '''
        Crossentropy
        or self-defined classes
        '''
        pass


    def load_dataset(self):
        '''
        or self-defined classes
        '''
        pass


    def train_model(self):
        '''
        _train_epoch
        logger
        Monitor
        '''
        pass
    
    def _train_epoch(self):
        '''
        standard one
        '''
        pass


    def predict(self):
        raise NotImplementedError('should be specified in individual model')


    def _cal_log_list(self):
        pass

    def need_log(self, epoch)->bool:
        return epoch in self._log_list


    def _set_model_directory(self):
        pass

    def _save_model(self):
        self._model_saved = True
        pass

    def _load_model(self):
        self._model_loaded = True
        pass


    def _get_option(self, mpath):
        if mpath == None:
            self._opt = BaseOption().parse()
        else:
            self._opt = self._load_json_as_argparse(mpath)


    def _save_opt_to_json(self, opt, path):
        json_dir = os.path.join(path, "opt.json")
        argparse_dict = vars(opt)
        with open(json_dir, 'w') as outfile:
            json.dump(argparse_dict, outfile)
        print ("configs have been dumped into %s" % json_dir)
        self._json_saved = True

    def _load_json_as_argparse(self, path):
        try:
            json_dir = os.path.join(path, "opt.json")
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
    def __init__(self):
        pass

    def monitor_epoch(self):
        pass



if __name__ == "__main__":
    pass