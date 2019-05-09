import torch
import numpy as np 
import measure
import utils
import os
from model import Model


class ComputeMI:
    def __init__(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
        # dataset
        # if self._opt.dataset == "MNIST":
        #     train_data, test_data = utils.get_mnist()
        #     self._train_set = torch.utils.data.DataLoader(train_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
        #     self._test_set = torch.utils.data.DataLoader(test_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
        #     self._initialize_model(dims = self._opt.layer_dims)
        #     print("MNIST experiment")

        train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
        test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
        self._train_set = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
        self._test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)


        # elif self._opt.dataset == "IBNet":
        #     train_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=True)
        #     test_data = utils.CustomDataset('2017_12_21_16_51_3_275766', train=False)
        #     self._train_set = torch.utils.data.DataLoader(train_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
        #     self._test_set = torch.utils.data.DataLoader(test_data, batch_size=self._opt.batch_size, shuffle=True, num_workers=self._opt.num_workers)
        #     self._initialize_model(dims = self._opt.layer_dims)
        #     print("IBnet experiment")
        # else:
        #     raise RuntimeError('Do not have {name} dataset, Please be sure to use the existing dataset'.format(name = dataset))

        self._model = Model(dims = [12, 12, 10, 7, 5, 4, 3, 2, 2], train = False)


        # self.FULL_MI = True
        # self.infoplane_measure = 'upper'
        # self.DO_SAVE        = True    # Whether to save plots or just show them
        # self.DO_LOWER       = (self.infoplane_measure == 'lower')   # Whether to compute lower bounds also
        # self.DO_BINNED      = (self.infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning
        # self.MAX_EPOCHS = 10000      # Max number of epoch for which to compute mutual information measure
        # self.NUM_LABELS = 2
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


    # def get_entropy_bound(self):
    #     Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
    #     entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
    #     entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])
    #     return entropy_func_upper, entropy_func_lower

    def get_saved_labelixs_and_labelprobs(self):
        saved_labelixs = {}
        y = self.tst.y
        Y = self.tst.Y
        if self.FULL_MI:
            full = utils.construct_full_dataset(self.trn, self.tst)
            y = full.y
            Y = full.Y
        for i in range(self.NUM_LABELS):
            saved_labelixs[i] = y == i
        labelprobs = np.mean(Y, axis=0)

        return saved_labelixs, labelprobs

    def computeMI(self):

        epoch_files = os.listdir('./results/IBNet_Time_05_08_14_58_Model_12_12_10_7_5_4_3_2_2_/')
        for epoch_file in epoch_files:

            ckpt = torch.load('./results/IBNet_Time_05_08_14_58_Model_12_12_10_7_5_4_3_2_2_/' + epoch_file)
            self._model.load_state_dict(ckpt['model_state_dict'])
            print(ckpt['model_state_dict'])
            epoch = ckpt['epoch']
            self._model.eval()

            layer_activity = []
            X = []
            Y = []
            for j, (inputs, labels) in enumerate(self._test_set):
                outputs = self._model(inputs)
                print(outputs[5])
                print(outputs[6])
                Y.append(labels)
                X.append(inputs)
                for i in range(len(outputs)):
                    data = outputs[i]
                    if len(layer_activity) < len(outputs):
                        layer_activity.append(data)
                    else:
                        layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)
            # print('-----------------------------')
            # print(layer_activity[6][101])
            # print('-----------------------------')
            
            for layer in layer_activity:
                # print('-----------------------------')
                # print(layer[100])
                # print('-----------------------------')
                test = self.measure.entropy_estimator_kl(layer, 0.001)
                # print(test)




        # entropy_func_upper, entropy_func_lower = self.get_entropy_bound()
        # saved_labelixs, labelprobs = self.get_saved_labelixs_and_labelprobs()
        
        # for activation in self.measures.keys():
        #     cur_dir = 'rawdata/' + self.DIR_TEMPLATE % activation
        #     if not os.path.exists(cur_dir):
        #         print("Directory %s not found" % cur_dir)
        #         continue
        #     # Load files saved during each epoch, and compute MI measures of the activity in that epoch
        #     print('*** Doing %s ***' % cur_dir)
        #     for epochfile in sorted(os.listdir(cur_dir)):
        #         if not epochfile.startswith('epoch'):
        #             continue
        #         fname = cur_dir + "/" + epochfile
        #         with open(fname, 'rb') as f:
        #             d = pickle.load(f)
        #         epoch = d['epoch']
        #         if epoch in self.measures[activation]: # Skip this epoch if its already been processed
        #             continue                      # this is a trick to allow us to rerun this cell multiple times)
        #         if epoch > self.MAX_EPOCHS:
        #             continue
        #         print("Doing", fname)
        #         num_layers = len(d['data']['activity_tst'])
        #         if self.PLOT_LAYERS is None:
        #             self.PLOT_LAYERS = []
        #             for lndx in range(num_layers):
        #                 #if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
        #                 self.PLOT_LAYERS.append(lndx)
        #         cepochdata = defaultdict(list)
        #         for lndx in range(num_layers):
        #             activity = d['data']['activity_tst'][lndx]
        #             # Compute marginal entropies
        #             h_upper = entropy_func_upper([activity,])[0]
        #             if self.DO_LOWER:
        #                 h_lower = entropy_func_lower([activity,])[0]
        #             # Layer activity given input. This is simply the entropy of the Gaussian noise
        #             hM_given_X = kde.kde_condentropy(activity, self.noise_variance)
        #             # Compute conditional entropies of layer activity given output
        #             hM_given_Y_upper=0.
        #             for i in range(self.NUM_LABELS):
        #                 hcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
        #                 hM_given_Y_upper += labelprobs[i] * hcond_upper
        #             if self.DO_LOWER:
        #                 hM_given_Y_lower=0.
        #                 for i in range(self.NUM_LABELS):
        #                     hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
        #                     hM_given_Y_lower += labelprobs[i] * hcond_lower
        #             cepochdata['MI_XM_upper'].append( self.nats2bits * (h_upper - hM_given_X) )
        #             cepochdata['MI_YM_upper'].append( self.nats2bits * (h_upper - hM_given_Y_upper) )
        #             cepochdata['H_M_upper'  ].append( self.nats2bits * h_upper )
        #             pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                    
        #             if self.DO_LOWER:  # Compute lower bounds
        #                 cepochdata['MI_XM_lower'].append(self.nats2bits * (h_lower - hM_given_X) )
        #                 cepochdata['MI_YM_lower'].append(self.nats2bits * (h_lower - hM_given_Y_lower) )
        #                 cepochdata['H_M_lower'  ].append(self.nats2bits * h_lower )
        #                 pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

        #             if self.DO_BINNED: # Compute binned estimates
        #                 binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, self.binsize)
        #                 cepochdata['MI_XM_bin'].append( self.nats2bits * binxm )
        #                 cepochdata['MI_YM_bin'].append( self.nats2bits * binym )
        #                 pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])
        #             print('- Layer %d %s' % (lndx, pstr) )
        #         self.measures[activation][epoch] = cepochdata

if __name__ == "__main__":
    t = ComputeMI()
    t.computeMI()