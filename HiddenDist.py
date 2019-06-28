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
from plot_utils import PlotFigure

class HiddenDist():
    def __init__(self, model_name = None, save_root = None):
        load_config = JsonParser() # training args
        # model_name = 'IBNet_test_plot_acc_loss_tanhx_Time_06_25_15_48'
        # save_root = './results'

        if model_name == None:
            if save_root == None:
                self.model_name, self.path = utils.find_newest_model('./results') # auto-find the newest model
            else:
                self.model_name, self.path = utils.find_newest_model(save_root)
        else:
            self.model_name = model_name
            if save_root == None:
                self.path = os.path.join('./results', self.model_name)
            else:
                self.path = os.path.join(save_root, self.model_name)

        print(self.model_name)

        self._opt = load_config.read_json_as_argparse(self.path) # load training args

        # force the batch size to 1 for calculation convinience
        self._opt.batch_size = 512

        train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
        test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
        # self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
        dataset = torch.utils.data.ConcatDataset([train_data, test_data])
        self._test_set = torch.utils.data.DataLoader(dataset, batch_size = self._opt.batch_size, shuffle = False, num_workers = 0)

        self._model = Model(opt = self._opt, train = False)

    def CalculateDist(self):
        model_path = os.path.join(self.path, 'models')
        epoch_files = os.listdir(model_path)

        # initialize plotter
        plotter = PlotFigure(self._opt, self.model_name, IS_HIDDEN_DIST=True)

        for epoch_file in epoch_files:
            if not epoch_file.endswith('.pth'):
                continue
            # load ckpt
            ckpt = torch.load(os.path.join(model_path, epoch_file))
            epoch = ckpt['epoch']

            #check if this epoch need to be calculated
            if not self.needLog(epoch):
                continue

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
                    data = outputs[i].detach().numpy()
                    if len(layer_activity) < len(outputs):
                        layer_activity.append(data)
                    else:
                        # layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)
                        layer_activity[i] = np.concatenate((layer_activity[i], data), axis = 0)

            # plot hidden output distribution for each epoch
            plotter.plot_hidden_dist(epoch, layer_activity)

        # generate gif for hidden output distribution
        plotter.generate_hidden_dist_gif()


    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert len(self._opt.log_seperator) == len(self._opt.log_frequency), "sha bi"
        for idx, val in enumerate(self._opt.log_seperator):
            if epoch < val:
                return epoch % self._opt.log_frequency[idx] == 0

if __name__ == "__main__":
    # act = HiddenDist()
    # act.CalculateDist()

    save_root = './results'
    model = 'IBNet_test_new_opt_tanhx_Time_06_27_18_24'
    # save_root = '/Users/xyli1905/Desktop/exp_ADAM'
    # model = None

    if model == None:
        for d in os.listdir(save_root):
            bd = os.path.join(save_root, d)
            if os.path.isdir(bd):
                hid = HiddenDist(model_name = d, save_root = save_root)
                hid.CalculateDist()
    else:
        hid = HiddenDist(model_name = model, save_root = save_root)
        hid.CalculateDist()

