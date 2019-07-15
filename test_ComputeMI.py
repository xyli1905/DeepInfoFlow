import torch
import numpy as np
import measure_utils as measure
import os
from SeqModel import SeqModel
import time
from plot_utils import PlotFigure
import threading
from dataLoader import DataProvider
from ModelInfoWrap import ModelInfo

class ComputeMI:
    @ModelInfo
    def __init__(self, model_name = None, save_root = None, measure_type = 'EVKL'):
        # ------------------------------------------------------------------ #
        # NOTE self.model_path and self.model_name from Decorator ModelInfo  #
        # ------------------------------------------------------------------ #
        self.progress_bar = 0

        # set model
        self._model = SeqModel(IS_TRAIN=False, model_path=self.model_path)
        self._opt = self._model.get_opt()

        # set dataset
        dataProvider = DataProvider(dataset_name = self._opt.dataset,  batch_size = self._opt.batch_size, num_workers = 0, shuffle = False)
        self.dataset = dataProvider.get_full_data()
        print("Measuring on ", self._opt.dataset)
        print("batch size: ", self._opt.batch_size)

        # get measure
        self.measure_type = measure_type


    def eval(self):
        if self.measure_type == 'EVKL':
            self.EVMethod()
        elif self.measure_type == 'kde':
            self.kdeMethod()


    def get_saved_labelixs_and_labelprobs(self):
        saved_labelixs = {}
        label_probs = []
        num_samples = 0
        # iter over dataset to get all labels
        for i, (data, label) in enumerate(self.dataset):
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
            t = threading.Thread(target=self.EVMethod)
            t.start()

    def random_index(self, size, max_size=4096):
        index_pairs = {
			"XT": np.random.choice(max_size, size, replace=False),   # indexes of P(X, T)
			"YT": np.random.choice(max_size, size, replace=False),   # indexes of P(Y, T)
			"X_XT" : np.random.choice(max_size, size, replace=False), # indexes of X in P(X)P(T)
			"T_XT" : np.random.choice(max_size, size, replace=False), # indexes of T in P(X)P(T)
			"Y_YT" : np.random.choice(max_size, size, replace=False), # indexes of Y in P(Y)P(T)
			"T_YT" : np.random.choice(max_size, size, replace=False)  # indexes of T in P(Y)P(T)
		}
        for idx in range(len(index_pairs["XT"][0])):
            ele1 = index_pairs['XT'][0,idx]
            ele2 = index_pairs["X_XT"][0,idx]
            ele3 = index_pairs["T_XT"][0,idx]
            print(f"{ele1}, {ele2}, {ele3}\n")

        return index_pairs

    # proposed method for empirical variational analysis
    def EVMethod(self):
        # start = time.time()
        print(f"calculation begins at {time.asctime()}")

        IX_dic = {}
        IY_dic = {}

        # prepare sample indices
        Nrepeats = 1
        random_indexes = self.random_index((Nrepeats, 1000))

        # container for activations, features and labels
        layer_activity = []
        X = np.array([])
        Y = np.array([])

        # inference on test set to get layer activations
        for j, (inputs, labels) in enumerate(self.dataset):
            # outputs = self._model.predict(inputs)
            epsilon = np.random.normal(0., 10., size = inputs.shape)
            e_t = torch.from_numpy(epsilon).float()
            outputs = self.func(inputs + e_t)
            np_labels = labels.clone().numpy().reshape(-1,1)
            np_inputs = inputs.clone().squeeze(0).numpy()
            X = np.vstack((X, np_inputs)) if len(X) != 0 else np_inputs
            Y = np.vstack((Y, np_labels)) if len(Y) != 0 else np_labels

            # for each layer activation add to container
            for i in range(len(outputs)):
                data = outputs[i]
                if len(layer_activity) < len(outputs):
                    layer_activity.append(data)
                else:
                    layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)


        norm_factor = np.log2(np.e)

        IX_epoch = []
        IY_epoch = []
        for layer in layer_activity:
            layer = layer.detach().numpy()

            avg_IX, avg_IY = self._compute_averaged_IX_IY(X, Y, layer, random_indexes)

        print(f"avg_IX = {avg_IX*norm_factor}; avg_IY = {avg_IY*norm_factor}")


    def _compute_averaged_IX_IY(self, X, Y, layer, random_indexes):
        Nrepeats = random_indexes["XT"].shape[0]

        avg_IX = 0.
        avg_IY = 0.
        for i in range(Nrepeats):
            # random sampling all the data
            XT_X = X[random_indexes["XT"][i]] # P(X,T) for X
            YT_Y = Y[random_indexes["YT"][i]] # P(Y,T) for Y
            XT_T = layer[random_indexes["XT"][i]] # P(X,T) for T
            YT_T = layer[random_indexes["YT"][i]] # P(Y,T) for T

            X_XT = X[random_indexes["X_XT"][i]] # P(X)(Y) for X
            Y_YT = Y[random_indexes["Y_YT"][i]] # P(Y)(T) for Y
            T_XT = layer[random_indexes["T_XT"][i]] # P(X)P(T) for T
            T_YT = layer[random_indexes["T_YT"][i]] # P(Y)P(T) for T

            # MI for X and T: I(X;T) = Dkl(P(X,T)||P(X)P(T))
            sample_XT_pair = np.concatenate((XT_X, XT_T), axis = 1)
            sample_X_and_T = np.concatenate((X_XT, T_XT), axis = 1)

            IX = measure.MI_estimator(sample_XT_pair, sample_X_and_T)
            avg_IX += IX

            # MI for Y and T: I(Y;T) = Dkl(P(Y,T)||P(Y)P(T))
            sample_YT_pair = np.concatenate((YT_Y, YT_T), axis = 1)
            sample_Y_and_T = np.concatenate((Y_YT, T_YT), axis = 1)

            IY = measure.MI_estimator(sample_YT_pair, sample_Y_and_T)
            avg_IY += IY

        return avg_IX / Nrepeats, avg_IY / Nrepeats
    
    def func(self, input):
        '''
        input: dim = (512, 12)
        '''
        # A = np.random.normal(loc=1.0, scale=1.0, size=(12, 10))
        dim = input.shape[1]
        A = np.ones(shape=(dim, dim))
        A_tensor = torch.from_numpy(A).float()
        return [torch.matmul(input, A_tensor)]

    def kdeMethod(self):
        pass





if __name__ == "__main__":

    measure_type = 'EVKL'
    t = ComputeMI(measure_type=measure_type)
    t.eval()

  