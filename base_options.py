# pass (basic) parameters to the model
import argparse
import os

class BaseOption:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # directory options

        # Arguments For IBnet Begin
        self._parser.add_argument('--batch_size', type=int, default=64, help='number of data points in one batch')
        self._parser.add_argument('--lr', type=float, default=0.04, help='learning rate')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        self._parser.add_argument('--max_epoch', type=int, default=8000, help='number of epochs')
        self._parser.add_argument('--num_workers', type=int, default=4, help='number of threads')
        self._parser.add_argument('--weight_decay', type=float, default=0.9, help='weight sdecay')

        self._parser.add_argument('--full_mi', type=self.boolean_string, default=True, help='weather construct full dataset')
        self._parser.add_argument('--activation', type=str, default='tanh', help='activation method')
        self._parser.add_argument('--save_root_dir', type=str, default='./results', help='directory to store outputs of evaluation of a model')
        self._parser.add_argument('--dataset', type=str, default='IBNet', help='dataset')

        self._parser.add_argument('--std', type=self.boolean_string, default=True, help='whether to save nets gradient standard deviation')
        self._parser.add_argument('--mean', type=self.boolean_string, default=True, help='whether to save nets gradient mean')
        self._parser.add_argument('--l2n', type=self.boolean_string, default=True, help='whether to save nets weight L2 normalization')



        # Arguments For IBnet End
        self._parser.add_argument('--chkp_dir', type=str, default='./checkpoints', help='directory storing trained models and optimizers')
        self._parser.add_argument('--data_dir', type=str, default='./data_proc/processed_data', help='directory storing preprocessed data')
        self._parser.add_argument('--results_dir', type=str, default='./results', help='directory to store outputs of evaluation of a model')


        # general options for training  (same for E and C)
        self._parser.add_argument('--is_train', type=self.boolean_string, default=True, help='flag showing if the model is in training')
        self._parser.add_argument('--is_debug', type=self.boolean_string, default=False, help='flags for debug mode')

        self._initialized = True

    def boolean_string(self, s):
        if s not in {'False', 'True', '0', '1'}:
            raise ValueError('Not a valid boolean string')
        return (s == 'True') or (s == '1')

    def parse(self):
        if not self._initialized:
            self.initialize()

        self._opt = self._parser.parse_args()


        # create results folder
        if not os.path.exists(self._opt.results_dir):
            os.makedirs(self._opt.results_dir)

        # create debug folder if need
        if self._opt.is_debug:
            print("running debuging mode")
            if not os.path.exists(self._opt.debug_dir):
                os.makedirs(self._opt.debug_dir)
        else:
            print("running normal mode")

        return self._opt