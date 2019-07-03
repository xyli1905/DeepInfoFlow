import utils
import os
import torch


class NumericalExperiment:
    def __init__(self, model_name = None, save_root = None):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setup
        self.model_name = model_name
        self.save_root = save_root
        self.manual_set_model_path()
        self._get_model_path()
    
    def manual_set_model_path(self):
        # l_name = 'IBNet_test_plot_acc_loss_tanhx_Time_06_25_15_48'
        # save_root = './results'
        pass

    def _get_model_path(self):
        if self.model_name == None:
            if self.save_root == None:
                self.model_name, self.model_path = utils.find_newest_model('./results') # auto-find the newest model
            else:
                self.model_name, self.model_path = utils.find_newest_model(self.save_root)
        else:
            if self.save_root == None:
                self.model_path = os.path.join('./results', self.model_name)
            else:
                self.model_path = os.path.join(self.save_root, self.model_name)
        
        print(self.model_name)