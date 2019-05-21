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
        self.weight_grad  = [] # for mean, std
        self.weight_value = [] # for l2n
        self.bias_grad    = [] # for mean, std
        self.bias_value   = [] # for l2n
        self.plotter = PlotFigure(self._opt, self.plot_name)
        self.recorded_epochs = []
    def createDataDict(self):
        layer_size = len(self._opt.layer_dims) - 1
        epoch_num  = self._opt.max_epoch
        source_keys  = ["weight", "bias"]
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
                    self.data["weight"]["mean"][epoch_key][layer_key] = self.dataParser( i, "mean", isWeight=True)
                    self.data["bias"]["mean"][epoch_key][layer_key] = self.dataParser( i, "mean", isWeight=False)
                if self._opt.std:
                    self.data["weight"]["std"][epoch_key][layer_key] = self.dataParser(i, "std", isWeight=True)
                    self.data["bias"]["std"][epoch_key][layer_key] = self.dataParser(i, "std", isWeight=False)
                if self._opt.l2n:
                    self.data["weight"]["l2n"][epoch_key][layer_key] = self.dataParser(i, "l2n", isWeight=True)
                    self.data["bias"]["l2n"][epoch_key][layer_key] = self.dataParser(i, "l2n", isWeight=False)

            # dump grad for debug
            # pickle.dump(name, f)
            # with open('grad.pkl', 'ab') as f:
            #     pickle.dump(self.weight_grad, f)
            # print(self.weight_grad)

            self.clear()

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
    def dataParser(self, layer, _type="mean", isWeight=True):
        if isWeight:
            grad, value = self.weight_grad[layer], self.weight_value[layer]
        else:
            grad, value = self.bias_grad[layer], self.bias_value[layer]

        if _type == "mean":
            grad = torch.reshape(grad, (grad.shape[0], -1))
            mean = torch.mean(grad, dim = 0)
            return torch.norm(mean).item()
        elif _type == "std":
            grad = torch.reshape(grad, (grad.shape[0], -1))
            # std = torch.std(grad, dim = 0)
            # return torch.norm(std).item()
            return torch.std(grad).item()
        elif _type == "l2n":
            return torch.norm(value).item()

    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert(len(self.log_seperator) == len(self.log_frequency), "sha bi")
        for idx, val in enumerate(self.log_seperator):
            if epoch < val:
                return epoch % self.log_frequency[idx] == 0

    def plot_mean_std(self):
        epoch_std = []
        epoch_mean = []
        for epoch in self.recorded_epochs:
            epoch_key = 'epoch' + str(epoch)
            layer_std = []
            layer_mean = []
            for layer in range(len(self._opt.layer_dims) - 1):
                layer_key = 'layer' + str(layer)
                layer_mean.append(self.data["weight"]["mean"][epoch_key][layer_key])
                layer_std.append(self.data["weight"]["std"][epoch_key][layer_key])
            epoch_mean.append(layer_mean)
            epoch_std.append(layer_std)
        epoch_mean = np.array(epoch_mean)
        epoch_std = np.array(epoch_std)

        self.plotter.plot_mean_std(self.recorded_epochs, epoch_mean, epoch_std)

    def __str__(self):
        pprint.pprint(self.data)
        return " "

