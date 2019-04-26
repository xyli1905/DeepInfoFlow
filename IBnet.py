from __future__ import print_function
import numpy as np
import os, pickle
from collections import defaultdict, OrderedDict

import simplebinmi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')

import torch
import torch.nn as nn
import torch.optim as optim

#import pytorch as pt
#import kde
#import utils
#import loggingreporter 


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.Dense1 = nn.Linear(12, 12)
        self.Dense2 = nn.Linear(12, 10)
        self.Dense3 = nn.Linear(10, 7)
        self.Dense4 = nn.Linear(7, 5)
        self.Dense5 = nn.Linear(5, 4)
        self.Dense6 = nn.Linear(4, 3)
        self.Dense7 = nn.Linear(3, 2)
        self.Dense8 = nn.Linear(2, 2)

    def forward(self,x):
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        x = self.Dense6(x)
        x = self.Dense7(x)
        x = self.Dense8(x)
        return x





class SaveActivations:
    def __init__(self):
        self.cfg = {}
        self.init_cfg()
        self.trn, self.tst = utils.get_IB_data('2017_12_21_16_51_3_275766')

    def init_cfg(self):
        self.cfg['BATCHSIZE']       = 256
        self.cfg['LEARNINGRATE']    = 0.0004
        self.cfg['MOMENTUM']        = 0.9
        self.cfg['NUM_EPOCHS']      = 8000
        self.cfg['FULL_MI']         = True
        self.cfg['ACTIVATION']      = 'tanh'
        # self.cfg['ACTIVATION']    = 'relu'
        # self.cfg['ACTIVATION']    = 'softsign'
        # self.cfg['ACTIVATION']    = 'softplus'
        self.cfg['LAYER_DIMS']      = [12,10,7,5,4,3,2] # original IB network
        ARCH_NAME =  '-'.join(map(str, self.cfg['LAYER_DIMS']))
        self.cfg['SAVE_DIR']        = 'rawdata/' + self.cfg['ACTIVATION'] + '_' + ARCH_NAME  # Where to save activation and weights data

    def initialize_model(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        model = Model()
        model.apply(weights_init)
        optimizer = optim.SGD(model.parameters(), 
                                lr=self.cfg['LEARNINGRATE'], 
                                momentum=self.cfg['MOMENTUM'])
        criterion = nn.CrossEntropyLoss() # loss


    def do_report(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        if epoch < 20:       # Log for all first 20 epochs
            return True
        elif epoch < 100:    # Then for every 5th epoch
            return (epoch % 5 == 0)
        elif epoch < 2000:    # Then every 10th
            return (epoch % 20 == 0)
        else:                # Then every 100th
            return (epoch % 100 == 0)
        
        reporter = loggingreporter.LoggingReporter(cfg=self.cfg, 
                                            trn=self.trn, 
                                            tst=self.tst, 
                                            do_save_func=self.do_report)



class ComputeMI:
    def __init__(self):
        self.trn, self.tst = utils.get_IB_data('2017_12_21_16_51_3_275766')
        self.FULL_MI = True
        self.infoplane_measure = 'upper'
        self.DO_SAVE        = True    # Whether to save plots or just show them
        self.DO_LOWER       = (self.infoplane_measure == 'lower')   # Whether to compute lower bounds also
        self.DO_BINNED      = (self.infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning
        self.MAX_EPOCHS = 10000      # Max number of epoch for which to compute mutual information measure
        self.NUM_LABELS = 2
        # self.MAX_EPOCHS = 1000
        self.COLORBAR_MAX_EPOCHS = 10000
        self.ARCH = '12-10-7-5-4-3-2'
        self.DIR_TEMPLATE = '%%s_%s'%self.ARCH
        self.noise_variance = 1e-3                    # Added Gaussian noise variance
        self.binsize = 0.07                           # size of bins for binning method
        self.nats2bits = 1.0/np.log(2) # nats to bits conversion factor
        self.PLOT_LAYERS    = None
        self.measures = OrderedDict()
        self.init_measures()

    def init_measures(self):
        
        self.measures['tanh'] = {}
        self.measures['relu'] = {}
        # self.measures['softsign'] = {}
        # self.measures['softplus'] = {}


    def get_entropy_bound(self):
        # Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
        # entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
        # entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])
        return entropy_func_upper, entropy_func_lower

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
        entropy_func_upper, entropy_func_lower = self.get_entropy_bound()
        saved_labelixs, labelprobs = self.get_saved_labelixs_and_labelprobs()
        
        for activation in self.measures.keys():
            cur_dir = 'rawdata/' + self.DIR_TEMPLATE % activation
            if not os.path.exists(cur_dir):
                print("Directory %s not found" % cur_dir)
                continue
            # Load files saved during each epoch, and compute MI measures of the activity in that epoch
            print('*** Doing %s ***' % cur_dir)
            for epochfile in sorted(os.listdir(cur_dir)):
                if not epochfile.startswith('epoch'):
                    continue
                fname = cur_dir + "/" + epochfile
                with open(fname, 'rb') as f:
                    d = pickle.load(f)
                epoch = d['epoch']
                if epoch in self.measures[activation]: # Skip this epoch if its already been processed
                    continue                      # this is a trick to allow us to rerun this cell multiple times)
                if epoch > self.MAX_EPOCHS:
                    continue
                print("Doing", fname)
                num_layers = len(d['data']['activity_tst'])
                if self.PLOT_LAYERS is None:
                    self.PLOT_LAYERS = []
                    for lndx in range(num_layers):
                        #if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                        self.PLOT_LAYERS.append(lndx)
                cepochdata = defaultdict(list)
                for lndx in range(num_layers):
                    activity = d['data']['activity_tst'][lndx]
                    # Compute marginal entropies
                    h_upper = entropy_func_upper([activity,])[0]
                    if self.DO_LOWER:
                        h_lower = entropy_func_lower([activity,])[0]
                    # Layer activity given input. This is simply the entropy of the Gaussian noise
                    hM_given_X = kde.kde_condentropy(activity, self.noise_variance)
                    # Compute conditional entropies of layer activity given output
                    hM_given_Y_upper=0.
                    for i in range(self.NUM_LABELS):
                        hcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
                        hM_given_Y_upper += labelprobs[i] * hcond_upper
                    if self.DO_LOWER:
                        hM_given_Y_lower=0.
                        for i in range(self.NUM_LABELS):
                            hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
                            hM_given_Y_lower += labelprobs[i] * hcond_lower
                    cepochdata['MI_XM_upper'].append( self.nats2bits * (h_upper - hM_given_X) )
                    cepochdata['MI_YM_upper'].append( self.nats2bits * (h_upper - hM_given_Y_upper) )
                    cepochdata['H_M_upper'  ].append( self.nats2bits * h_upper )
                    pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                    
                    if self.DO_LOWER:  # Compute lower bounds
                        cepochdata['MI_XM_lower'].append(self.nats2bits * (h_lower - hM_given_X) )
                        cepochdata['MI_YM_lower'].append(self.nats2bits * (h_lower - hM_given_Y_lower) )
                        cepochdata['H_M_lower'  ].append(self.nats2bits * h_lower )
                        pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

                    if self.DO_BINNED: # Compute binned estimates
                        binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, self.binsize)
                        cepochdata['MI_XM_bin'].append( self.nats2bits * binxm )
                        cepochdata['MI_YM_bin'].append( self.nats2bits * binym )
                        pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])
                    print('- Layer %d %s' % (lndx, pstr) )
                self.measures[activation][epoch] = cepochdata
