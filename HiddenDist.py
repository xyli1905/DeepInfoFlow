import torch
import numpy as np
import os
from SeqModel import SeqModel
import time
from plot_utils import PlotFigure
from dataLoader import DataProvider
from ModelInfoWrap import ModelInfo

class HiddenDist:
    @ModelInfo
    def __init__(self, model_name = None, save_root = None):
        # ------------------------------------------------------------------ #
        # NOTE self.model_path and self.model_name from Decorator ModelInfo  #
        # ------------------------------------------------------------------ #     
        # set model
        self._model = SeqModel(IS_TRAIN=False, model_path=self.model_path)
        self._opt = self._model.get_opt()

        # force the batch size to 1 for calculation convinience
        self._opt.batch_size = 512
        
        # set dataset
        dataProvider = DataProvider(dataset_name = self._opt.dataset,  batch_size = self._opt.batch_size, num_workers = 0, shuffle = False)
        self.dataset = dataProvider.get_full_data()
        print("Measuring on ", self._opt.dataset)
        print("batch size: ", self._opt.batch_size)

    def CalculateDist(self):
        print(f"calculation begins at {time.asctime()}")
        
        ckpt_path = os.path.join(self.model_path, self._opt.ckpt_dir)
        epoch_files = os.listdir(ckpt_path)

        # initialize plotter
        plotter = PlotFigure(self._opt, self.model_name, IS_HIDDEN_DIST=True)

        for epoch_file in epoch_files:
            if not epoch_file.endswith('.pth'):
                continue

            # load model epoch weight
            indicators = self._model.load_model(epoch_file, CKECK_LOG=True)
            if not indicators["NEED_LOG"]:
                continue # if this epoch does not need to be logged continue
            epoch = indicators["epoch"]
            # set model to eval
            self._model.eval()

            # container for activations, features and labels
            layer_activity = []

            # inference on test set to get layer activations
            for j, (inputs, labels) in enumerate(self.dataset):
                outputs = self._model.predict(inputs)
                # for each layer activation add to container
                for i in range(len(outputs)):
                    data = outputs[i].detach().numpy()
                    if len(layer_activity) < len(outputs):
                        layer_activity.append(data)
                    else:
                        # layer_activity[i] = torch.cat((layer_activity[i], data), dim = 0)
                        layer_activity[i] = np.concatenate((layer_activity[i], data), axis = 0)

            # plot hidden output distribution for each epoch
            plotter.plot_dist(epoch, layer_activity, plot_type='hidden')

        # generate gif for hidden output distribution
        plotter.generate_dist_gif(plot_type='hidden')



if __name__ == "__main__":
    # act = HiddenDist()
    # act.CalculateDist()

    save_root = './results'
    model_name = 'IBNet_test_new_opt_tanhx_Time_06_27_18_24'
    # save_root = '/Users/xyli1905/Desktop/exp_ADAM'
    # model_name = None
    
    if model_name == None:
        for d in os.listdir(save_root):
            bd = os.path.join(save_root, d)
            if os.path.isdir(bd):
                hid = HiddenDist(model_name = d, save_root = save_root)
                hid.CalculateDist()
    else:
        hid = HiddenDist(model_name = model_name, save_root = save_root)
        hid.CalculateDist()

