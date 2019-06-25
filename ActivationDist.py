import torch
import numpy as np
import measure
import utils
import os
from model import Model
from json_parser import JsonParser
import time
from plot_utils import PlotFigure
import sys
import threading
import pickle
import matplotlib
import matplotlib.pyplot as plt

class ActivationDist():
    def __init__(self):
        load_config = JsonParser() # training args
        self.model_name = None

        self.model_name = 'IBNet_IB_net_test_3_Time_05_28_04_44_Model_12_10_7_5_4_3_2_'
        self.path = os.path.join('./results', self.model_name)

        if self.model_name == None:
            self.model_name, self.path = utils.find_newest_model('./results') # auto-find the newest model
        print(self.model_name)
        self._opt = load_config.read_json_as_argparse(self.path) # load training args

        # force the batch size to 1 for calculation convinience
        self._opt.batch_size = 1

        train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
        test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
        # self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
        dataset = torch.utils.data.ConcatDataset([train_data, test_data])
        self._test_set = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        self._model = Model(activation = self._opt.activation ,dims = self._opt.layer_dims, train = False)

    def CalculateDist(self):
        # epoch_files = os.listdir(self.path)
        # for epoch_file in epoch_files:
        #     if not epoch_file.endswith('.pth'):
        #         continue
        #     # load ckpt
        #     ckpt = torch.load(os.path.join(self.path, epoch_file))
        #     epoch = ckpt['epoch']

        #     #check if this epoch need to be calculated
        #     if not self.needLog(epoch):
        #         continue

        #     # load model epoch weight
        #     self._model.load_state_dict(ckpt['model_state_dict'])
        #     # set model to eval
        #     self._model.eval()

        #     # container for activations, features and labels
        #     layer_activity = []

        #     # inference on test set to get layer activations
        #     for j, (inputs, labels) in enumerate(self._test_set):
        #         outputs = self._model(inputs)
        #         # for each layer activation add to container
        #         for i in range(len(outputs)):
        #             data = outputs[i]
        #             if len(layer_activity) < len(outputs):
        #                 layer_activity.append(data)
        #             else:
        #                 layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)


        #     figure, axs = plt.subplots(3, 2, sharey=True, tight_layout=True)
        #     axs = axs.flatten()
        #     ims = []
        #     for i in range(len(layer_activity)):
        #         data = layer_activity[i].reshape(-1)
        #         data = data.detach().numpy()
        #         axs[i].hist(data, bins = 300)
        #     plt.show()


        ckpt = torch.load(os.path.join(self.path, "model_epoch_4000.pth"))
        # load model epoch weight
        self._model.load_state_dict(ckpt['model_state_dict'])
        # set model to eval
        self._model.eval()

        # container for activations, features and labels
        layer_activity = []

        # inference on test set to get layer activations
        for j, (inputs, labels) in enumerate(self._test_set):
            outputs = self._model(inputs)
            # for each layer activation add to container
            for i in range(len(outputs)):
                data = outputs[i]
                if len(layer_activity) < len(outputs):
                    layer_activity.append(data)
                else:
                    layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)


        figure, axs = plt.subplots(1, 6, sharey=True, tight_layout=True)
        ims = []
        for i in range(len(layer_activity)):
            data = layer_activity[i].reshape(-1)
            data = data.detach().numpy()
            axs[i].hist(data, bins = 50)
        plt.show()



    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert len(self._opt.log_seperator) == len(self._opt.log_frequency), "sha bi"
        for idx, val in enumerate(self._opt.log_seperator):
            if epoch < val:
                return epoch % self._opt.log_frequency[idx] == 0

if __name__ == "__main__":
    act = ActivationDist()
    act.CalculateDist()

