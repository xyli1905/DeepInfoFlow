import torch
import pprint
from plot_utils import PlotFigure
import numpy as np 
import pickle
import os
import re

class Logger(object):
    def __init__(self, opt, plot_name):
        self._opt = opt
        self.plot_name = plot_name
        self.data = self.createDataDict()

        self.weight_grad  = [] # to store W'
        self.weight_value = [] # to store W
        self.bias_grad    = [] # to store bias'
        self.bias_value   = [] # to store bias

        self.full_epoch_list = [] # record whole list of epochs
        self.loss      = [] # to record loss
        self.acc_train = [] # to record training accuracy
        self.acc_test  = [] # to record test accuracy

        self.plotter = PlotFigure(self._opt, self.plot_name)
        self.recorded_epochs = []

        self.svds = [[], []] # first for original SVD and second for normalized SVD

        self._process_param_names = re.compile(r"^D.+(weight|bias)$")

        self.layer_weight_mean = []

        self.weight_dist_images = []
        self.grad_dist_images = []

    def createDataDict(self):
        layer_size = len(self._opt.layer_dims) - 1
        epoch_num  = self._opt.max_epoch
        source_keys  = ["weight_value" ,"weight_grad", "bias", "bias_grad"]
        type_keys    = ["mean", "std", "l2n"]
        epoch_keys   = list(map(lambda x: "epoch" + str(x), [i for i in range(epoch_num)]))
        layer_keys   = list(map(lambda x: "layer" + str(x), [i for i in range(layer_size)]))
        data = {}
        for source_key in source_keys:
            if source_key not in data.keys():
                data[source_key] = {}
                for type_key in type_keys:
                    if type_key not in data[source_key].keys():
                        data[source_key][type_key] = {}
                        for epoch_key in epoch_keys:
                            if epoch_key not in data[source_key][type_key].keys():
                                data[source_key][type_key][epoch_key] = {}
                                for layer_key in layer_keys:
                                    if layer_key not in data[source_key][type_key][epoch_key].keys():
                                        data[source_key][type_key][epoch_key][layer_key] = 0
        return data


    def log(self, model):
        # record model parameters
        if len(self.weight_grad) == 0 and len(self.weight_value) == 0 and len(self.bias_grad) == 0 and len(self.bias_value) == 0:
            for name, param in model.named_parameters():
                if self._process_param_names.match(name):
                    grad = param.grad.clone().detach().unsqueeze(0)
                    data = param.data.clone().detach().unsqueeze(0)
                    if name.endswith('weight'):
                        self.weight_grad.append(grad)
                        self.weight_value.append(data)
                    if name.endswith('bias'):
                        self.bias_grad.append(grad)
                        self.bias_value.append(data)
        else:
            index = 0
            for name, param in model.named_parameters():
                if self._process_param_names.match(name):
                    grad = param.grad.clone().detach().unsqueeze(0)
                    data = param.data.clone().detach().unsqueeze(0)
                    if name.endswith('weight'):
                        self.weight_grad[index] = torch.cat((self.weight_grad[index], grad), dim = 0)
                        self.weight_value[index] = torch.cat((self.weight_value[index], data), dim = 0)
                    if name.endswith('bias'):
                        self.bias_grad[index] = torch.cat((self.bias_grad[index], grad), dim = 0)
                        self.bias_value[index] = torch.cat((self.bias_value[index], data), dim = 0)
                        index += 1
        
    def log_acc_loss(self, epoch, record_type, acc, loss=None):
        if record_type == 'train':
            # record acc and loss for training process
            self.acc_train.append(acc)
            self.loss.append(loss)
            self.full_epoch_list.append(epoch) # append epoch only for train case
        elif record_type == 'test':
            # record acc for training process
            self.acc_test.append(acc)
        else:
            raise ValueError('not valid record type')


    def update(self, epoch):
        self.recorded_epochs.append(epoch)
        epoch_key = "epoch" + str(epoch)
        for i in range(len(self.weight_grad)):
            layer_key = "layer" + str(i)
            if self._opt.mean:
                self.data["weight_value"]["mean"][epoch_key][layer_key] = self.dataParser( i, "mean", isWeight=True, isGrad=False)
                self.data["weight_grad"]["mean"][epoch_key][layer_key] = self.dataParser( i, "mean", isWeight=True, isGrad=True)
                self.data["bias"]["mean"][epoch_key][layer_key] = self.dataParser( i, "mean", isWeight=False, isGrad= False)
                self.data["bias_grad"]["mean"][epoch_key][layer_key] = self.dataParser( i, "mean", isWeight=False, isGrad= True)
            if self._opt.std:
                self.data["weight_value"]["std"][epoch_key][layer_key] = self.dataParser( i, "std", isWeight=True, isGrad=False)
                self.data["weight_grad"]["std"][epoch_key][layer_key] = self.dataParser( i, "std", isWeight=True, isGrad=True)
                self.data["bias"]["std"][epoch_key][layer_key] = self.dataParser( i, "std", isWeight=False, isGrad= False)
                self.data["bias_grad"]["std"][epoch_key][layer_key] = self.dataParser( i, "std", isWeight=False, isGrad= True)
            if self._opt.l2n:
                self.data["weight_value"]["l2n"][epoch_key][layer_key] = self.dataParser( i, "l2n", isWeight=True, isGrad=False)
                self.data["weight_grad"]["l2n"][epoch_key][layer_key] = self.dataParser( i, "l2n", isWeight=True, isGrad=True)
                self.data["bias"]["l2n"][epoch_key][layer_key] = self.dataParser( i, "l2n", isWeight=False, isGrad= False)
                self.data["bias_grad"]["l2n"][epoch_key][layer_key] = self.dataParser( i, "l2n", isWeight=False, isGrad= True)
        self.plot_dist_epoch(epoch)
        self.calculate_svd()
        self.clear()

    def dataParser(self, layer, _type="mean", isWeight=True, isGrad = True, method = 1):
        if isWeight and isGrad:
            tensor = self.weight_grad[layer]
        elif isWeight and not isGrad:
            tensor = self.weight_value[layer]
        elif not isWeight and isGrad:
            tensor = self.bias_grad[layer]
        elif not isWeight and not isGrad:
            tensor = self.bias_value[layer]
        else:
            raise RuntimeError('error in calculate weight and gradient data')

        if _type == "mean":
            if method == 1:
                ##METHOD 1: batch-averaged then take norm for each layer
                mean = torch.mean(tensor, dim = 0)
                if isWeight and not isGrad:
                    self.layer_weight_mean.append(mean)
                return torch.norm(mean).item()
            if method == 2:
                ##METHOD 2: averge tha (abs) of all w in a layer in a epoch
                mean = torch.mean(tensor.abs())
                # mean = torch.mean(tensor)
                return mean.item()
            if method == 3:
                ##METHOD 3: average within each layer, then norm along batch
                reshaped_tensor = torch.reshape(tensor, (tensor.shape[0], -1))
                mean = torch.mean(reshaped_tensor, dim = 1)
                return torch.norm(mean).item()
        elif _type == "std":
            # reshaped_tensor = torch.reshape(tensor, (tensor.shape[0], -1))
            # std = torch.std(reshaped_tensor, dim = 0)
            if method == 1:
                ##METHOD 1: std along batches then take norm for each layer
                std = torch.std(tensor, dim = 0)
                return torch.norm(std).item()
            if method == 2:
                ##METHOD 2: std of the average of tha abs of all w in a layer in a epoch
                mean = torch.mean(tensor, dim = 0)
                return torch.std(mean).item()
            if method == 3:
                ##METHOD 3: cal. std within each layer, then norm along batch
                reshaped_tensor = torch.reshape(tensor, (tensor.shape[0], -1))
                std = torch.std(reshaped_tensor, dim = 1)
                return torch.norm(std).item()
        elif _type == "l2n":
            return torch.norm(tensor).item()
        else:
            raise RuntimeError('error in calculate weight and gradient data')

    def calculate_svd(self):
        one_epoch_original_weight = []
        one_epoch_normalized_weight = []
        # one_epoch_grad = []

        # for calculating weight svd
        for i in range(len(self.weight_value)):
            mean_weight = self.layer_weight_mean[i]
            _, weight_sigma, _ = torch.svd(mean_weight, compute_uv = False)
            ##NOTE either use the original s or the normalized one
            weight_sigma_tmp = weight_sigma.numpy()
            one_epoch_normalized_weight.append(weight_sigma_tmp/weight_sigma_tmp[0])
            one_epoch_original_weight.append(weight_sigma_tmp)
            # print(one_epoch_weight)
        self.svds[0].append(one_epoch_original_weight) # [Lepoch] [NLayer] [weight_layers]
        self.svds[1].append(one_epoch_normalized_weight) # [Lepoch] [NLayer] [weight_layers]
        ##NOTE svd for grad, note used presently
        # for calcularing grad svd
        # for grad in self.weight_grad:
        #     mean_grad = torch.mean(grad, dim = 0)
        #     _, grad_sigma, _ = torch.svd(mean_grad, compute_uv = False)
        #     one_epoch_grad.append(grad_sigma.numpy())
        # self.svds[1].append(one_epoch_grad)
        #######################################

    def clear(self):
        self.weight_grad = []
        self.weight_value = []
        self.bias_grad = []
        self.bias_value = []
        self.layer_weight_mean = []


    def plot_figures(self, mean_and_std = True, sv = True, acc_loss = True, dist_gif = True):
        # save epoch data
        if mean_and_std or sv or acc_loss or dist_gif:
            # print(self.recorded_epochs)
            self.plotter.save_plot_data("recorded_epochs_data.pkl", self.recorded_epochs)
        # plot mean_and_std and save the data
        if mean_and_std:
            epoch_mean, epoch_std = self.get_mean_std()
            self.plotter.plot_mean_std(self.recorded_epochs, epoch_mean, epoch_std)
            self.plotter.save_plot_data("mean_data.pkl", epoch_mean)
            self.plotter.save_plot_data("std_data.pkl", epoch_std)
        # plot sv and save the data
        if sv:
            self.plotter.plot_svd(self.recorded_epochs, self.svds)
            self.plotter.save_plot_data("svds_data.pkl", self.svds)
        # plot acc_loss and save the data
        if acc_loss:
            self.plotter.plot_acc_loss(self.full_epoch_list, self.acc_train, self.acc_test, self.loss)
            self.plotter.save_plot_data("full_epoch_list_data.pkl", self.full_epoch_list)
            self.plotter.save_plot_data("acc_train_data.pkl", self.acc_train)
            self.plotter.save_plot_data("acc_test_data.pkl", self.acc_test)
            self.plotter.save_plot_data("loss_data.pkl", self.loss)
        # generate gif for dist plot of weight and weight grad
        if dist_gif:
            # self.plotter.generate_dist_gif(plot_type = 'weight')
            # self.plotter.generate_dist_gif(plot_type = 'grad')
            import threading
            threads = []
            t1 = threading.Thread(target=self.plotter.generate_dist_gif_by_images, args = ('weight', self.weight_dist_images))
            threads.append(t1)
            t2 = threading.Thread(target=self.plotter.generate_dist_gif_by_images, args = ('grad', self.grad_dist_images))
            threads.append(t2)
            # self.plotter.generate_dist_gif_by_images(plot_type = 'weight', images = self.weight_dist_images)
            # self.plotter.generate_dist_gif_by_images(plot_type = 'grad', images = self.grad_dist_images)
            for t in threads:
                t.setDaemon(True)
                t.start()

            for i in threads:
                i.join()

    def plot_dist_epoch(self, epoch):
        mean_weight, mean_grad = self.cal_batch_mean()
        weight_image = self.plotter.plot_dist(epoch, mean_weight, plot_type='weight')
        grad_image = self.plotter.plot_dist(epoch, mean_grad, plot_type='grad')
        self.weight_dist_images.append(weight_image)
        self.grad_dist_images.append(grad_image)


    def cal_batch_mean(self):
        mean_weight = []
        mean_grad = []
        for layer in range(len(self.weight_value)):
            tmp_weight = self.layer_weight_mean[layer].numpy()
            tmp_grad   = torch.mean(self.weight_grad[layer], dim = 0).numpy()
            mean_weight.append(tmp_weight)
            mean_grad.append(tmp_grad)

        return mean_weight, mean_grad

    def get_mean_std(self):
        epoch_std = []
        epoch_mean = []
        for epoch in self.recorded_epochs:
            epoch_key = 'epoch' + str(epoch)
            layer_std = []
            layer_mean = []
            for layer in range(len(self._opt.layer_dims) - 1):
                layer_key = 'layer' + str(layer)
                layer_mean.append(self.data["weight_grad"]["mean"][epoch_key][layer_key])
                layer_std.append(self.data["weight_grad"]["std"][epoch_key][layer_key])
            epoch_mean.append(layer_mean)
            epoch_std.append(layer_std)
        epoch_mean = np.array(epoch_mean)
        epoch_std = np.array(epoch_std)

        return epoch_mean, epoch_std


    def __str__(self):
        pprint.pprint(self.data)
        return " "

