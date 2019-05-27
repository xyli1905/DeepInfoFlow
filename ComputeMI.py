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
        self.model_name = 'IBNet_test_EVKL_Time_05_27_13_19_Model_12_12_10_7_5_4_3_2_2_'
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
            t = threading.Thread(target=self.EVMethod)
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

        IX_dic = {}
        IY_dic = {}

        Nrepeats = 1
        random_indexes = self.random_index((Nrepeats, 500))

        print("len dataset : ", len(self._test_set))
        epoch_files = os.listdir(self.path)
        for epoch_file in epoch_files:
            # random_indexes = self.random_index(4096)

            progress += 1
            # random_sampled_points = {}

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
                Y.append(labels.clone().numpy())
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

            IX_epoch = []
            IY_epoch = []
            for layer in layer_activity:
                layer = layer.detach().numpy()

                avg_IX, avg_IY = self._compute_averaged_IX_IY(X, Y, layer, random_indexes)
                
                IX_epoch.append(avg_IX)
                IY_epoch.append(avg_IY)

            if epoch not in IX_dic.keys() and epoch not in IY_dic.keys():
                IX_dic[epoch] = IX_epoch
                IY_dic[epoch] = IY_epoch
            else:
                raise RuntimeError('epoch is duplicated')

        plotter = PlotFigure(self._opt, self.model_name)
        plotter.plot_MI_plane(IX_dic, IY_dic)
        end = time.time()
        print(" ")
        print("total time cost : ", end - start)

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

            # print("------"*5)
            # # print(random_indexes["XT"])
            # # print(" ")
            # print(repr(random_indexes["X_XT"]))
            # print(" ")
            # print(repr(random_indexes["T_XT"]))
            # print(" ")
            # print(repr(random_indexes["Y_YT"]))
            # print(" ")
            # print(repr(random_indexes["T_YT"]))
            # print("------"*5)

            # MI for X and T: I(X;T) = Dkl(P(X,T)||P(X)P(T))
            sample_XT_pair = np.concatenate((XT_X, XT_T), axis = 1)
            sample_X_and_T = np.concatenate((X_XT, T_XT), axis = 1)

            IX = self.measure.MI_estimator(sample_XT_pair, sample_X_and_T)
            avg_IX += IX

            # MI for Y and T: I(Y;T) = Dkl(P(Y,T)||P(Y)P(T))
            sample_YT_pair = np.concatenate((YT_Y, YT_T), axis = 1)
            sample_Y_and_T = np.concatenate((Y_YT, T_YT), axis = 1)

            IY = self.measure.MI_estimator(sample_YT_pair, sample_Y_and_T)
            avg_IY += IY

        return avg_IX / Nrepeats, avg_IY / Nrepeats


    def kdeMethod(self):
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

        plotter = PlotFigure(self._opt, self.model_name)
        plotter.plot_MI_plane(IX, IY)
        end = time.time()
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
    # t.kdeMethod()
    t.EVMethod()

    # a = np.array([1883, 2775, 2152, 1959, 1411,  552, 3765,  899,  903, 3711, 3298,
    #    2801,  154,  101, 2656, 2468,  525, 2159, 1857, 1434, 1726, 4094,
    #    3554,  178, 1346, 1954, 1719, 3952,   88, 3234, 3434, 1441, 3409,
    #    1922, 4081, 4049, 2831,  399, 2396, 1092,  248, 1599, 3146, 1008,
    #    2368, 1705, 1317, 3958, 3247,  407, 1423, 2145, 1251, 1711, 3578,
    #    2899, 3782, 3081,  345,  787, 1039, 3812, 3061, 3840, 2730, 3330,
    #     895, 2607, 3415, 2611, 3623, 2717, 1945,  394,  261, 2933,  149,
    #     847,  722,  105, 3957,  807, 4091, 3644, 2287,   31, 3074,  412,
    #    2577, 2055, 3069, 3957, 2496, 1847,  800,  357, 2938, 3142, 2446,
    #    2362])

    # b = np.array([3033, 1355, 2119, 2657,  302,  767, 2488, 2444, 3565, 1934, 2357,
    #     660, 2686, 1422, 2787, 1284, 2831,  864,  888, 1205,  887, 3900,
    #    2613, 1061, 1507, 2300, 1445, 1489, 2236, 3586,  322, 1024, 2914,
    #    3447,  677, 2033, 3400, 4068,  257, 3043,  172, 2442,  279, 3028,
    #    2483, 4013, 2893, 3760, 3675, 3388, 2981, 1564, 3198, 2368, 3024,
    #    3591, 2967, 3939, 1141,  555,  843, 1401,  894, 1995, 2214, 3214,
    #    1730, 2156, 1683,  794,  948, 2931, 3384, 3785, 2753, 2569,  319,
    #    3841, 4013,  790, 3453,   41, 2307, 2554, 3036, 2193, 2168, 3777,
    #     626, 1863, 2876, 3422, 1515, 1079,  996,  286,  385,  841,  653,
    #    1924])

    # c = np.array([ 333, 2835, 2673,  951,  116, 2479, 4078, 1378, 2173, 1181, 2284,
    #    1132, 2631,  692, 3970, 1861, 1595,  478,   63, 2534,  285, 1770,
    #    3114, 1794, 2286, 2295, 3079,  365, 1321, 3461, 1802,  209, 3555,
    #     388,  144, 2169, 3995, 1293, 1924, 2584, 2415,  987, 3234, 1446,
    #    3243, 2690,  170, 3384, 2999, 2158,   90, 1762,  979,  495, 1152,
    #    1875, 3125, 3135, 1246, 3400, 2493, 2982, 1032,  116, 3407,  766,
    #     192,  473,   89, 2793, 1998, 3420, 2788, 3355,  631, 1435, 2609,
    #     535, 1746, 1147,  581, 3042, 3499, 1305,  428, 1144,   54, 2809,
    #    2490, 3369, 1746, 3609, 1067, 2126, 1544, 3136,  687,  557, 2019,
    #     435])

    # d = np.array([  54, 1630, 2260, 4057, 4027, 3905, 1172, 2652,  626, 1762, 1600,
    #    3231, 2546, 4047,  896,  793,  142, 4010, 2932, 2398, 1972,  140,
    #     404, 3372, 3715, 2781, 3586,  827, 3030,  669,  606, 1186, 3567,
    #    2499, 2370, 3276, 3959, 3665,   31, 2826, 3251,  961, 1824,  386,
    #     236,  839, 1344, 4001, 2108, 1945, 1746, 4019, 3400,  654, 2205,
    #     304, 3977, 3446, 3002,   99,  536, 1561,  971, 1986, 1258, 3089,
    #    1061, 1349, 4070,  281, 2435, 2771, 2455, 3905, 2467,  858, 1756,
    #    3215, 2729, 1828, 1990, 1098, 1577,  981, 3375, 3188,   81, 2032,
    #    3408,   42, 3867, 3815, 1551, 3892, 3687, 2182, 3896, 3499, 3610,
    #    2595])

    def check(a, b, c, d):
        a = list(a)
        b = list(b)
        c = list(c)
        d = list(d)

        repeat_a = [(i, j) for i in range(len(a)) for j in range(len(a)) if a[i] == a[j] and i < j]
        repeat_b = [(i, j) for i in range(len(b)) for j in range(len(b)) if b[i] == b[j] and i < j]
        repeat_c = [(i, j) for i in range(len(c)) for j in range(len(c)) if c[i] == c[j] and i < j]
        repeat_d = [(i, j) for i in range(len(d)) for j in range(len(d)) if d[i] == d[j] and i < j]
        print (repeat_a)
        print (repeat_b)
        print (repeat_c)
        print (repeat_d)