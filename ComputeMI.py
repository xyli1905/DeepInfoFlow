import torch
import numpy as np 
import measure
import utils
import os
from model import Model
from json_parser import JsonParser
import time
from plot_utils import PlotFigure


class ComputeMI:
    def __init__(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
        load_config = JsonParser() # training args
        self.path =os.path.join('./results', 'IBNet_IB_net_test_1__Time_05_09_21_31_Model_12_12_10_7_5_4_3_2_2_')# info plane dir
        self._opt = load_config.read_json_as_argparse(self.path) # load training args

        # force the batch size to 1 for calculation convinience
        self._opt.batch_size = 1
        # dataset
        if self._opt.dataset == "MNIST":
            train_data, test_data = utils.get_mnist()
            self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
            self._test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
            print("MNIST experiment")

        elif self._opt.dataset == "IBNet":
            train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
            test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
            self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
            self._test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
            print("IBnet experiment")
        else:
            raise RuntimeError('Do not have {name} dataset, Please be sure to use the existing dataset'.format(name = self._opt.dataset))
        
        # get model
        self._model = Model(dims = self._opt.layer_dims, train = False)
        # get measure
        self.measure = measure.kde()


    def get_saved_labelixs_and_labelprobs(self):
        saved_labelixs = {}
        label_probs = []
        num_samples = 0
        # iter over dataset to get all labels
        for i, (data, label) in enumerate(self._test_set):
            if label.item() not in saved_labelixs.keys():
                saved_labelixs[label.item()] = [i]
            else:
                saved_labelixs[label.item()].append(i)
            num_samples = i+1
        # calculate label prob
        for key in sorted(saved_labelixs.keys()):
            label_probs.append(len(saved_labelixs[key]) / num_samples)


        return saved_labelixs, label_probs

    def computeMI(self):
        saved_labelixs, label_probs = self.get_saved_labelixs_and_labelprobs()

        epoch_files = os.listdir(self.path)
        start = time.time()

        IX = {}
        IY = {}

        nats2bits = 1.0/np.log(2)

        for epoch_file in epoch_files:
            if not epoch_file.endswith('.pth'):
                continue

            # load ckpt
            ckpt = torch.load(os.path.join(self.path, epoch_file))
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
            X = []
            Y = []

            # inference on test set to get layer activations
            for j, (inputs, labels) in enumerate(self._test_set):
                outputs = self._model(inputs)
                Y.append(labels)
                X.append(inputs)

                # for each layer activation add to container
                for i in range(len(outputs)):
                    data = outputs[i]
                    if len(layer_activity) < len(outputs):
                        layer_activity.append(data)
                    else:
                        layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)
            # for each layer compute IX and IY
            IX_epoch = []
            IY_epoch = [] 
            for layer in layer_activity:
                upper = self.measure.entropy_estimator_kl(layer, 0.001)
                hM_given_X = self.measure.kde_condentropy(layer, 0.001)

                mutual_info_X = upper - hM_given_X # IX
                IX_epoch.append(mutual_info_X.item() * nats2bits)

                # for each label y
                hM_given_Y_upper=0.
                for i, key in enumerate(sorted(saved_labelixs.keys())):
                    hcond_upper = self.measure.entropy_estimator_kl(layer[saved_labelixs[key]], 0.001)
                    hM_given_Y_upper += label_probs[i] * hcond_upper 

                mutual_info_Y = upper - hM_given_Y_upper
                IY_epoch.append(mutual_info_Y.item() * nats2bits)

            assert len(IX_epoch) == 8, "layer dims error"
            assert len(IY_epoch) == 8, "layer dims error"

            
            if epoch not in IX.keys() and epoch not in IY.keys():
                IX[epoch] = IX_epoch
                IY[epoch] = IY_epoch
            else:
                raise RuntimeError('epoch is duplicated')

        end = time.time()
        plotter = PlotFigure(self._opt)
        plotter.plot_MI_plane(IX, IY)
        print(end - start)


    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert len(self._opt.log_seperator) == len(self._opt.log_frequency), "sha bi"
        for idx, val in enumerate(self._opt.log_seperator):
            if epoch < val:
                return epoch % self._opt.log_frequency[idx] == 0


if __name__ == "__main__":
    t = ComputeMI()
    t.computeMI()