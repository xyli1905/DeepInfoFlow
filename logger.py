import torch

class Logger(object):
    def __init__(self, opt):
        self._opt = opt
        self.log_seperator = self._opt.log_seperator
        self.log_frequency = self._opt.log_frequency
        self.data = self.createDataDict()
        self.weight_grad  = [[] for i in range(len(opt.layer_dims) - 1)] # for mean, std
        self.weight_value = [[] for i in range(len(opt.layer_dims) - 1)] # for l2n
        self.bias_grad    = [[] for i in range(len(opt.layer_dims) - 1)] # for mean, std
        self.bias_value   = [[] for i in range(len(opt.layer_dims) - 1)] # for l2n
        
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
            epoch_key = "epoch" + str(epoch)
            for i in range(len(self.weight_grad)):
                layer_key = "layer" + str(i)
                if self._opt.mean:
                    self.data["weight"]["mean"][epoch_key][layer_key] = self.dataParser("mean", isWeight=True)
                    self.data["bias"]["mean"][epoch_key][layer_key] = self.dataParser("mean", isWeight=False)
                if self._opt.std:
                    self.data["weight"]["std"][epoch_key][layer_key] = self.dataParser("std", isWeight=True)
                    self.data["bias"]["std"][epoch_key][layer_key] = self.dataParser("std", isWeight=False)
                if self._opt.l2n:
                    self.data["weight"]["l2n"][epoch_key][layer_key] = self.dataParser("l2n", isWeight=True)
                    self.data["bias"]["l2n"][epoch_key][layer_key] = self.dataParser("l2n", isWeight=False)

    def log(self, model):
        for i, (name, param) in enumerate(model.named_parameters()):
            for i in range(len(self.weight_grad)):
                self.weight_grad[i].
            
    def dataParser(self, _type="mean", isWeight=True):
        
        if isWeight:
            grad, value = self.weight_grad, self.weight_value
        else:
            grad, value = self.bias_grad, self.bias_value

        if _type == "mean":
            return 1
        elif _type == "std":
            return 2
        elif _type == "l2n":
            return 3

    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert(len(self.log_seperator) == len(self.log_frequency), "sha bi")
        for idx, val in enumerate(self.log_seperator):
            if epoch < val:
                return epoch % self.log_frequency[idx] == 0

