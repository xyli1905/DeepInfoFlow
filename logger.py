from base_options import BaseOption
import re

class Logger(object):
    def __init__(self):
        self._opt = BaseOption().parse()
        self.log_seperator = self._opt.log_seperator
        self.log_frequency = self._opt.log_frequency
        self.data = self.createDataDict()
        
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

    def log(self, model, epoch):
        if self.needLog(epoch):
            epoch_key = "epoch" + str(epoch)
            for i, (name, params) in enumerate(model.named_parameters()):
                matrix = params.grad # use to calculate mean, std, l2n
                
                if name.endswith("weight"):
                    layer_key = "layer" + re.findall(".*D.(.*).weight.*", name)[0]
                    if self._opt.mean:
                        self.data["weight"]["mean"][epoch_key][layer_key] = int(re.findall(".*D.(.*).weight.*", name)[0]) * int(re.findall(".*D.(.*).weight.*", name)[0])
                    if self._opt.std:
                        self.data["weight"]["std"][epoch_key][layer_key] = int(re.findall(".*D.(.*).weight.*", name)[0]) * int(re.findall(".*D.(.*).weight.*", name)[0])
                    if self._opt.l2n:
                        self.data["weight"]["l2n"][epoch_key][layer_key] = int(re.findall(".*D.(.*).weight.*", name)[0]) * int(re.findall(".*D.(.*).weight.*", name)[0])

                elif name.endswith("bias"):
                    layer_key = "layer" + re.findall(".*D.(.*).bias.*", name)[0]
                    if self._opt.mean:
                        self.data["bias"]["mean"][epoch_key][layer_key] = int(re.findall(".*D.(.*).bias.*", name)[0]) * int(re.findall(".*D.(.*).bias.*", name)[0])
                    if self._opt.std:
                        self.data["bias"]["std"][epoch_key][layer_key] = int(re.findall(".*D.(.*).bias.*", name)[0]) * int(re.findall(".*D.(.*).bias.*", name)[0])
                    if self._opt.l2n:
                        self.data["bias"]["l2n"][epoch_key][layer_key] = int(re.findall(".*D.(.*).bias.*", name)[0]) * int(re.findall(".*D.(.*).bias.*", name)[0])
            


    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert(len(self.log_seperator) == len(self.log_frequency), "sha bi")
        for idx, val in enumerate(self.log_seperator):
            if epoch < val:
                return epoch % self.log_frequency[idx] == 0

