import torch
import numpy as np 
import measure
import utils
import os
from model import Model
from json_parser import JsonParser
import time


class ComputeMI:
    def __init__(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
        load_config = JsonParser()
        self.path ='./results/' + 'IBNet_Time_05_09_18_05_Model_12_12_10_7_5_4_3_2_2_' + '/'
        self._opt = load_config.read_json_as_argparse(self.path)

        # force the batch size to 1 for calculation convinience
        self._opt.batch_size = 1
        # dataset
        if self._opt.dataset == "MNIST":
            train_data, test_data = utils.get_mnist()
            self._train_set = torch.utils.data.DataLoader(train_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            self._test_set = torch.utils.data.DataLoader(test_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            print("MNIST experiment")

        # train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
        # test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
        # self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
        # self._test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)


        elif self._opt.dataset == "IBNet":
            train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
            test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
            self._train_set = torch.utils.data.DataLoader(train_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            self._test_set = torch.utils.data.DataLoader(test_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
            print("IBnet experiment")
        else:
            raise RuntimeError('Do not have {name} dataset, Please be sure to use the existing dataset'.format(name = dataset))

        self._model = Model(dims = self._opt.layer_dims, train = False)


        # self.FULL_MI = True
        # self.infoplane_measure = 'upper'
        # self.DO_SAVE        = True    # Whether to save plots or just show them
        # self.DO_LOWER       = (self.infoplane_measure == 'lower')   # Whether to compute lower bounds also
        # self.DO_BINNED      = (self.infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning
        # self.MAX_EPOCHS = 10000      # Max number of epoch for which to compute mutual information measure
        self.NUM_LABELS = 2
        # # self.MAX_EPOCHS = 1000
        # self.COLORBAR_MAX_EPOCHS = 10000
        # self.ARCH = '12-10-7-5-4-3-2'
        # self.DIR_TEMPLATE = '%%s_%s'%self.ARCH
        # self.noise_variance = 1e-3                    # Added Gaussian noise variance
        # self.binsize = 0.07                           # size of bins for binning method
        # self.nats2bits = 1.0/np.log(2) # nats to bits conversion factor
        # self.PLOT_LAYERS    = None

        self.results = {}
        self.measure = measure.kde()


    def get_saved_labelixs_and_labelprobs(self):
        saved_labelixs = {}
        labelprobs = []
        num_samples = 0
        for i, (data, label) in enumerate(self._test_set):
            if label.item() not in saved_labelixs.keys():
                saved_labelixs[label.item()] = [i]
            else:
                saved_labelixs[label.item()].append(i)
            num_samples = i+1

        for key in sorted(saved_labelixs.keys()):
            labelprobs.append(len(saved_labelixs[key]) / num_samples)


        return saved_labelixs, labelprobs

    def computeMI(self):
        saved_labelixs, labelprobs = self.get_saved_labelixs_and_labelprobs()

        epoch_files = os.listdir(self.path)
        start = time.time()
        for epoch_file in epoch_files:
            if not epoch_file.endswith('.pth'):
                continue
            ckpt = torch.load(self.path + epoch_file)
            self._model.load_state_dict(ckpt['model_state_dict'])
            epoch = ckpt['epoch']
            self._model.eval()

            layer_activity = []
            X = []
            Y = []
            for j, (inputs, labels) in enumerate(self._test_set):
                outputs = self._model(inputs)
                Y.append(labels)
                X.append(inputs)
                for i in range(len(outputs)):
                    data = outputs[i]
                    if len(layer_activity) < len(outputs):
                        layer_activity.append(data)
                    else:
                        layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)
            for layer in layer_activity:
                upper = self.measure.entropy_estimator_kl(layer, 0.001)
                hM_given_X = self.measure.kde_condentropy(layer, 0.001)
                # print(upper - hM_given_X)


                hM_given_Y_upper=0.
                for i, key in enumerate(sorted(saved_labelixs.keys())):
                    hcond_upper = self.measure.entropy_estimator_kl(layer[saved_labelixs[key]], 0.001)
                    hM_given_Y_upper += labelprobs[i] * hcond_upper
                # print(upper - hM_given_Y_upper)
            # print('------------------------------epoch {epoch}------------------------------'.format(epoch = epoch))
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    t = ComputeMI()
    t.computeMI()
    # t.get_saved_labelixs_and_labelprobs()