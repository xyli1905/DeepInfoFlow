# pass (basic) parameters to the model
import argparse
import os

class BaseOption:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # Arguments For IBnet Begin
        self._parser.add_argument('--experiment_name', type=str, default='test_new_code', help='a unique name for experiment')
        self._parser.add_argument('--layer_dims', type=list, default=[12, 10, 7, 5, 4, 3, 2], help='dimention of each layer')
        self._parser.add_argument('--dataset', type=str, default='IBNet', help='dataset')

        self._parser.add_argument('--activation', type=str, default='tanhx', help='activation method')
        #below four only apply to 'relux' and 'tanhx' cases
        self._parser.add_argument('--Vmax', type=float, default=None, help='Max Value for activationX')
        self._parser.add_argument('--Vmin', type=float, default=None, help='Min Value for activationX')
        self._parser.add_argument('--slope', type=float, default=1.0, help='slope for activationX')
        self._parser.add_argument('--dispX', type=float, default=0.0, help='x displacement for activationX')

        self._parser.add_argument('--batch_size', type=int, default=512, help='number of data points in one batch')
        self._parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs')
        self._parser.add_argument('--num_workers', type=int, default=0, help='number of threads')

        self._parser.add_argument('--optimizer', type=str, default='sgd', help='choice of optimizer')
        self._parser.add_argument('--lr', type=float, default=0.08, help='learning rate')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        self._parser.add_argument('--weight_decay', type=float, default=0.9, help='weight decay')

        self._parser.add_argument('--lossfunc', type=str, default='crossentropy', help='choice of loss function')
        
        # presently not used
        # self._parser.add_argument('--full_mi', type=self.boolean_string, default=True, help='weather construct full dataset')

        self._parser.add_argument('--std', type=self.boolean_string, default=True, help='whether to save nets gradient standard deviation')
        self._parser.add_argument('--mean', type=self.boolean_string, default=True, help='whether to save nets gradient mean')
        self._parser.add_argument('--l2n', type=self.boolean_string, default=True, help='whether to save nets weight L2 normalization')

        self._parser.add_argument('--log_seperator', type=list, default=[20, 100, 2000, 10000], help='number of epochs to change log frequency')
        self._parser.add_argument('--log_frequency', type=list, default=[1, 5, 20, 100], help='log frequency')


        # Arguments For IBnet End
        self._parser.add_argument('--save_root', type=str, default='./results', help='Path to store outputs of evaluation of a model')
        self._parser.add_argument('--data_path', type=str, default='./data_proc/processed_data', help='Path storing preprocessed data')
        self._parser.add_argument('--plot_path', type=str, default='./plots', help='Path to store outputs of information plane plots')
        self._parser.add_argument('--ckpt_dir', type=str, default='models', help='checkpoint directory storing trained models and optimizers')


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
        if not os.path.exists(self._opt.save_root):
            os.makedirs(self._opt.save_root)

        # # create debug folder if need
        # if self._opt.is_debug:
        #     print("running debuging mode")
        #     if not os.path.exists(self._opt.debug_dir):
        #         os.makedirs(self._opt.debug_dir)
        # else:
        #     print("running normal mode")

        return self._opt