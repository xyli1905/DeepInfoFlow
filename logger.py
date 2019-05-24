import torch
import pprint
from plot_utils import PlotFigure
import numpy as np 
import pickle

class Logger(object):
    def __init__(self, opt, plot_name):
        self._opt = opt
        self.plot_name = plot_name
        self.log_seperator = self._opt.log_seperator
        self.log_frequency = self._opt.log_frequency
        self.data = self.createDataDict()

        self.weight_grad  = [] # to store W'
        self.weight_value = [] # to store W
        self.bias_grad    = [] # to store bias'
        self.bias_value   = [] # to store bias

        self.plotter = PlotFigure(self._opt, self.plot_name)
        self.recorded_epochs = []

        self.svds = [[], []] # first for weight and second for grad
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

    def update(self, epoch):
        if self.needLog(epoch):
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
            self.calculate_svd()
        self.clear()
    def calculate_svd(self):
        one_epoch_weight = []
        one_epoch_grad = []
        # for calculating weight svd
        for weight in self.weight_value:
            mean_weight = torch.mean(weight, dim = 0)
            _, weight_sigma, _ = torch.svd(mean_weight, compute_uv = False)
            one_epoch_weight.append(weight_sigma)
        self.svds[0].append(one_epoch_weight)
        # for calcularing grad svd
        for grad in self.weight_grad:
            mean_grad = torch.mean(grad, dim = 0)
            _, grad_sigma, _ = torch.svd(mean_grad, compute_uv = False)
            one_epoch_grad.append(grad_sigma)
        self.svds[1].append(one_epoch_grad)

    def clear(self):
        self.weight_grad = []
        self.weight_value = []
        self.bias_grad = []
        self.bias_value = []

    def log(self, model):
        if len(self.weight_grad) == 0 and len(self.weight_value) == 0 and len(self.bias_grad) == 0 and len(self.bias_value) == 0:
            for name, param in model.named_parameters():
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
                grad = param.grad.clone().detach().unsqueeze(0)
                data = param.data.clone().detach().unsqueeze(0)
                if name.endswith('weight'):
                    self.weight_grad[index] = torch.cat((self.weight_grad[index], grad), dim = 0)
                    self.weight_value[index] = torch.cat((self.weight_value[index], data), dim = 0)
                if name.endswith('bias'):
                    self.bias_grad[index] = torch.cat((self.bias_grad[index], grad), dim = 0)
                    self.bias_value[index] = torch.cat((self.bias_value[index], data), dim = 0)
                    index += 1
    def dataParser(self, layer, _type="mean", isWeight=True, isGrad = True):
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
            reshaped_tensor = torch.reshape(tensor, (tensor.shape[0], -1))
            mean = torch.mean(reshaped_tensor, dim = 0)
            return torch.norm(mean).item()
        elif _type == "std":
            reshaped_tensor = torch.reshape(tensor, (tensor.shape[0], -1))
            std = torch.std(reshaped_tensor, dim = 0)
            return torch.norm(std).item()
        elif _type == "l2n":
            return torch.norm(tensor).item()
        else:
            raise RuntimeError('error in calculate weight and gradient data')

    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert len(self.log_seperator) == len(self.log_frequency), "sha bi"
        for idx, val in enumerate(self.log_seperator):
            if epoch < val:
                return epoch % self.log_frequency[idx] == 0

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
        # self.plotter.plot_mean_std(self.recorded_epochs, epoch_mean, epoch_std)

    def plot_figures(self, mean_and_std = True, svd = True):
        if mean_and_std:
            epoch_mean, epoch_std = self.get_mean_std()
            self.plotter.plot_mean_std(self.recorded_epochs, epoch_mean, epoch_std)
        if svd:
            ##############################################
            #to do: add method in plot_utils to plot svds#
            ##############################################
            pass


    def __str__(self):
        pprint.pprint(self.data)
        return " "

