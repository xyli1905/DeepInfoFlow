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

        # Arguments For IBnet End

        self._parser.add_argument('--chkp_dir', type=str, default='./checkpoints', help='directory storing trained models and optimizers')
        self._parser.add_argument('--data_dir', type=str, default='./data_proc/processed_data', help='directory storing preprocessed data')
        self._parser.add_argument('--results_dir', type=str, default='./results', help='directory to store outputs of evaluation of a model')

        # options for debug
        # - note that when used for test mode, it means to dump the index of wrong predict for further investigation
        self._parser.add_argument('--is_debug', type=self.boolean_string, default=False, help='flag for debug')
        self._parser.add_argument('--debug_dir', type=str, default='./debug', help='name dir that stores debug outputs')

        # model options
        self._parser.add_argument('--model_type', type=str, default='baseline', help='type of model: baseline, rnn, encoder-decoder')
        self._parser.add_argument('--classifier_net', type=str, default='LSTM', help='name of the classifier network used')
        self._parser.add_argument('--encoder_net', type=str, default='', help='name of the encoder network used')

        # RNN options
        self._parser.add_argument('--number_layers', type=int, default=1, help='number of RNN layers')
        self._parser.add_argument('--is_bidirectional', type=self.boolean_string, default=False, help='whether be bidirectional')
        self._parser.add_argument('--dropout_rate', type=float, default=0, help='dropout rate for output of RNN')

        # data options
        self._parser.add_argument('--train_data_name', type=str, default='train_mat.pkl', help='name for training data')
        self._parser.add_argument('--test_data_name', type=str, default='test_mat.pkl', help='name for test data')
        self._parser.add_argument('--vocab_name', type=str, default='vocab.pkl', help='file name for processed vocabulary')
        self._parser.add_argument('--pretrained_weight_name', type=str, default='glove.pkl', help='file name for processed pretrained weight')
        self._parser.add_argument('--number_workers', type=int, default=0, help='number of workers in DataLoader')

        # general options for training  (same for E and C)
        self._parser.add_argument('--is_train', type=self.boolean_string, default=True, help='flag showing if the model is in training')
        self._parser.add_argument('--is_emb_trainable', type=self.boolean_string, default=False, help='whether allow update pretrained embedding')

        # training options for classifier (& normal model)
        self._parser.add_argument('--load_epoch_C', type=int, default=0, help='idx of epoch for loading classifier')
        self._parser.add_argument('--max_epoch_C', type=int, default=1, help='number of epochs for training classifier')
        self._parser.add_argument('--lr_C', type=float, default=0.0001, help='learning rate for classifier')

        # training options for encoder
        self._parser.add_argument('--load_epoch_E', type=int, default=0, help='idx of epoch for loading encoder')
        self._parser.add_argument('--max_epoch_E', type=int, default=6, help='number of epochs for training encoder')
        self._parser.add_argument('--lr_E', type=float, default=0.04, help='learning rate for encoder')

        # option for balanced data
        self._parser.add_argument('--is_balanced', type=self.boolean_string, default=False, help='whether use balanced subset of dataset')

        # options for triplet loss
        self._parser.add_argument('--is_triplet', type=self.boolean_string, default=False, help='whether use triplet loss in training')
        self._parser.add_argument('--margin', type=float, default=0.01, help='margin in triplet loss')
        self._parser.add_argument('--iter_size', type=int, default=100000, help='number of triplets in a epoch')
        self._parser.add_argument('--round_num', type=int, default=1, help='number of rounds to train encoder and classifier')

        # for local test we take the last 'valid_num' sentences as the validation set
        self._parser.add_argument('--valid_num', type=int, default=100000, help='size of validation set')

        # options for save and display (same for C and E)
        self._parser.add_argument('--tag', type=str, default="null", help='tag for distinguishing the saved files')
        self._parser.add_argument('--save_freq', type=int, default=1, help='frequency (/epoch) for saving model')
        self._parser.add_argument('--loss_check_freq', type=int, default=-1, help='frequency (/iters) for outputing loss')
        self._parser.add_argument('--max_loss_check', type=int, default=10, help='upper bound for number of loss check')

        # options for test
        self._parser.add_argument('--threshold', type=float, default=0.5, help='threshold for classification')

        self._initialized = True

    def boolean_string(self, s):
        if s not in {'False', 'True', '0', '1'}:
            raise ValueError('Not a valid boolean string')
        return (s == 'True') or (s == '1')

    def parse(self):
        if not self._initialized:
            self.initialize()

        # self._opt = self._parser.parse_args(args=[]) # for use in jupyter
        self._opt = self._parser.parse_args()

        # save args to file
        args = vars(self._opt)
        self._save(args)

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

    def _save(self, args):
        expr_dir = os.path.join(self._opt.chkp_dir, self._opt.model_type)

        #prepare saving directory
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'option_list.txt' )
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')