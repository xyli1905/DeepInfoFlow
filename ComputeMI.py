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

class ComputeMI:
    def __init__(self):
        self.progress_bar = 0
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
        load_config = JsonParser() # training args
        self.model_name = 'IBNet_test_normalizedSVD_Time_05_24_16_49_Model_12_12_10_7_5_4_3_2_2_'
        self.path =os.path.join('./results', self.model_name)# info plane dir
        self._opt = load_config.read_json_as_argparse(self.path) # load training args

        # force the batch size to 1 for calculation convinience
        self._opt.batch_size = 1
        # dataset
        if self._opt.dataset == "MNIST":
            train_data, test_data = utils.get_mnist()

            if not self._opt.full_mi:
                # self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
                self._test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
            else:
                dataset = torch.utils.data.ConcatDataset([train_data, test_data])
                self._test_set = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            print("MNIST experiment")

        elif self._opt.dataset == "IBNet":
            train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
            test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
            # self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
            if not self._opt.full_mi:
                self._test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
            else:
                dataset = torch.utils.data.ConcatDataset([train_data, test_data])
                self._test_set = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            print("IBnet experiment")
        else:
            raise RuntimeError('Do not have {name} dataset, Please be sure to use the existing dataset'.format(name = self._opt.dataset))

        # get model
        self._model = Model(activation = self._opt.activation ,dims = self._opt.layer_dims, train = False)
        # get measure
        # self.measure = measure.kde()
        self.measure = measure.EVKL() # our new measure


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


    def launch_computeMI_Thread(self):
            t = threading.Thread(target=self.computeMI)
            t.start()

    def random_index(self, size, max_size=4096):
        index_pairs = {
			"XT": np.random.choice(max_size, size),   # indexes of P(X, T)
			"YT": np.random.choice(max_size, size),   # indexes of P(Y, T)
			"X_XT" : np.random.choice(max_size, size), # indexes of X in P(X)P(T)
			"T_XT" : np.random.choice(max_size, size), # indexes of T in P(X)P(T)
			"Y_YT" : np.random.choice(max_size, size), # indexes of Y in P(Y)P(T)
			"T_YT" : np.random.choice(max_size, size)  # indexes of T in P(Y)P(T)
		}
        return index_pairs

    # proposed method for empirical variational analysis
    def EVMethod(self):
        start = time.time()

        progress = 0

        epoch_files = os.listdir(self.path)
        for epoch_file in epoch_files:

            progress += 1
            random_indexes = self.random_index(10)

            random_sampled_points = {}

            self.progress_bar = int(str(round(float(progress / len(epoch_files)) * 100.0)))
            print("\rprogress : " + str(round(float(progress / len(epoch_files)) * 100.0)) + "%",end = "", flush = True)
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
                Y.append(labels.clone().squeeze(0).numpy())
                X.append(inputs.clone().squeeze(0).numpy())

                # for each layer activation add to container
                for i in range(len(outputs)):
                    data = outputs[i]
                    if len(layer_activity) < len(outputs):
                        layer_activity.append(data)
                    else:
                        layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)

            X = np.array(X)
            Y = np.array(Y)

            for layer in layer_activity:
                layer = layer.detach().numpy()

                # random sampling all the data
                XT_X = X[random_indexes["XT"]] # P(X,T) for X
                YT_Y = Y[random_indexes["YT"]] # P(Y,T) for Y
                XT_T = layer[random_indexes["XT"]] # P(X,T) for T
                YT_T = layer[random_indexes["YT"]] # P(Y,T) for T

                X_XT = X[random_indexes["X_XT"]] # P(X)(Y) for X
                Y_YT = Y[random_indexes["Y_YT"]] # P(Y)(T) for Y
                T_XT = layer[random_indexes["T_XT"]] # P(X)P(T) for T
                T_YT = layer[random_indexes["T_YT"]] # P(Y)P(T) for T

                # MI for X and T: I(X;T) = Dkl(P(X,T)||P(X)P(T))
                sample_XT_pair = np.concatenate((XT_X, XT_T), axis = 1)
                sample_X_and_T = np.concatenate((X_XT, T_XT), axis = 1)

                IX = self.measure.MI_estimator(sample_XT_pair, sample_X_and_T)

                # MI for Y and T: I(Y;T) = Dkl(P(Y,T)||P(Y)P(T))
                sample_YT_pair = np.concatenate((YT_Y, YT_T), axis = 1)
                sample_Y_and_T = np.concatenate((Y_YT, T_YT), axis = 1)

                IY = self.measure.MI_estimator(sample_YT_pair, sample_Y_and_T)



        end = time.time()
        print(" ")
        print("total time cost : ", end - start)


    def computeMI(self):
        saved_labelixs, label_probs = self.get_saved_labelixs_and_labelprobs()

        epoch_files = os.listdir(self.path)
        start = time.time()

        IX = {}
        IY = {}

        nats2bits = 1.0/np.log(2)

        progress = 0
        for epoch_file in epoch_files:
            progress += 1
            self.progress_bar = int(str(round(float(progress / len(epoch_files)) * 100.0)))
            print("\rprogress : " + str(round(float(progress / len(epoch_files)) * 100.0)) + "%",end = "", flush = True)
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

            if epoch not in IX.keys() and epoch not in IY.keys():
                IX[epoch] = IX_epoch
                IY[epoch] = IY_epoch
            else:
                raise RuntimeError('epoch is duplicated')

        end = time.time()
        plotter = PlotFigure(self._opt, self.model_name)
        plotter.plot_MI_plane(IX, IY)
        print(" ")
        print("total time cost : ", end - start)


    def needLog(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        assert len(self._opt.log_seperator) == len(self._opt.log_frequency), "sha bi"
        for idx, val in enumerate(self._opt.log_seperator):
            if epoch < val:
                return epoch % self._opt.log_frequency[idx] == 0


if __name__ == "__main__":
    t = ComputeMI()
    # t.computeMI()
    t.EVMethod()