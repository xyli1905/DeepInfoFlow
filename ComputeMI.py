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
        print(f"calculation begins at {time.asctime()}")

        IX_dic = {}
        IY_dic = {}

        # prepare sample indices
        Nrepeats = 1
        random_indexes = self.random_index((Nrepeats, 1000))

        print("len dataset : ", len(self.dataset.dataset))
        ckpt_path = os.path.join(self.model_path, self._opt.ckpt_dir)
        epoch_files = os.listdir(ckpt_path)
        num_epoch_files = len(epoch_files)

        progress = 0
        for epoch_file in epoch_files:
            if not epoch_file.endswith('.pth'):
                continue

            # running progress record
            progress += 1
            progress_ratio = float(progress / num_epoch_files) * 100.0
            # self.progress_bar = int(progress_ratio)
            print(f"\rprogress : {progress_ratio:.4f}%",end = "", flush = True)


            # load model epoch weight
            indicators = self._model.load_model(epoch_file, CKECK_LOG=True)
            if not indicators["NEED_LOG"]:
                continue # if this epoch does not need to be logged continue
            epoch = indicators["epoch"]
            # set model to eval
            self._model.eval()

            # container for activations, features and labels
            layer_activity = []
            X = np.array([])
            Y = np.array([])

            # inference on test set to get layer activations
            for j, (inputs, labels) in enumerate(self.dataset):
                outputs = self._model.predict(inputs)
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

        # save data, then plot
        plotter = PlotFigure(self._opt, self.model_name)
        plotter.save_plot_data("IX_dic_data.pkl", IX_dic)
        plotter.save_plot_data("IY_dic_data.pkl", IY_dic)
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



    def kdeMethod(self):
        start = time.time()
        
        saved_labelixs, label_probs = self.get_saved_labelixs_and_labelprobs()

        ckpt_path = os.path.join(self.model_path, self._opt.ckpt_dir)
        epoch_files = os.listdir(ckpt_path)
        num_epoch_files = len(epoch_files)

        IX = {}
        IY = {}

        nats2bits = 1.0/np.log(2)

        progress = 0
        for epoch_file in epoch_files:
            if not epoch_file.endswith('.pth'):
                continue

            progress += 1
            progress_ratio = float(progress / num_epoch_files) * 100.0
            # self.progress_bar = int(progress_ratio)
            print(f"\rprogress : {progress_ratio:.4f}%",end = "", flush = True)
            
            # load model epoch weight
            indicators = self._model.load_model(epoch_file, CKECK_LOG=True)
            if not indicators["NEED_LOG"]:
                continue # if this epoch does not need to be logged continue
            epoch = indicators["epoch"]
            # set model to eval
            self._model.eval()

            # container for activations, features and labels
            layer_activity = []
            X = []
            Y = []

            # inference on test set to get layer activations
            for j, (inputs, labels) in enumerate(self.dataset):
                outputs = self._model.predict(inputs)
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
                upper = measure.entropy_estimator_kl(layer, 0.001)
                hM_given_X = measure.kde_condentropy(layer, 0.001)

                mutual_info_X = upper - hM_given_X # IX
                IX_epoch.append(mutual_info_X.item() * nats2bits)

                # for each label y
                hM_given_Y_upper=0.
                for i, key in enumerate(sorted(saved_labelixs.keys())):
                    hcond_upper = measure.entropy_estimator_kl(layer[saved_labelixs[key]], 0.001)
                    hM_given_Y_upper += label_probs[i] * hcond_upper 

                mutual_info_Y = upper - hM_given_Y_upper
                IY_epoch.append(mutual_info_Y.item() * nats2bits)

            if epoch not in IX.keys() and epoch not in IY.keys():
                IX[epoch] = IX_epoch
                IY[epoch] = IY_epoch
            else:
                raise RuntimeError('epoch is duplicated')

        # save data, then plot
        plotter = PlotFigure(self._opt, self.model_name)
        plotter.save_plot_data("IX_dic_data.pkl", IX)
        plotter.save_plot_data("IY_dic_data.pkl", IY)
        plotter.plot_MI_plane(IX, IY)
        end = time.time()
        print(" ")
        print("total time cost : ", end - start)



if __name__ == "__main__":

    measure_type = 'EVKL'
    t = ComputeMI(measure_type=measure_type)
    t.eval()

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

    # def check(a, b, c, d):
    #     a = list(a)
    #     b = list(b)
    #     c = list(c)
    #     d = list(d)

    #     repeat_a = [(i, j) for i in range(len(a)) for j in range(len(a)) if a[i] == a[j] and i < j]
    #     repeat_b = [(i, j) for i in range(len(b)) for j in range(len(b)) if b[i] == b[j] and i < j]
    #     repeat_c = [(i, j) for i in range(len(c)) for j in range(len(c)) if c[i] == c[j] and i < j]
    #     repeat_d = [(i, j) for i in range(len(d)) for j in range(len(d)) if d[i] == d[j] and i < j]
    #     print (repeat_a)
    #     print (repeat_b)
    #     print (repeat_c)
    #     print (repeat_d)