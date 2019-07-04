import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cycler import cycler
import moviepy.editor as mpy
import numpy as np
import pickle
import datetime
import os
import utils
import sys
from ModelInfoWrap import ModelInfo

np.random.seed()


'''NOTE
Seem for conda env, one must create a file `matplotlibrc` at the directory `~/.matplotlib`,
and add the following content in this file:
                                            backend : TKAgg
see for reference: https://github.com/matplotlib/matplotlib/issues/13414
'''

class PlotFigure:
    @ModelInfo
    def __init__(self, opt=None, model_name=None, save_root = None, IS_HIDDEN_DIST=False):
        # ------------------------------------------------------------------ #
        # NOTE self.model_path and self.model_name from Decorator ModelInfo  #
        # ------------------------------------------------------------------ #
        self.name = 'Plot_Utils'

        if opt == None:
            self._opt = utils.load_json_as_argparse(self.model_path)
        else:
            self._opt = opt

        # NOTE we save figures in two places: the plot_root and results/model_path
        # check existence of plot_root
        self.plot_path = self._opt.plot_path
        if not os.path.exists(self._opt.plot_path):
            os.mkdir(self._opt.plot_path)

        # model_path must exist already
        self.model_plot_fig_path = os.path.join(self.model_path, 'plots_fig')
        if not os.path.exists(self.model_plot_fig_path):
            os.mkdir(self.model_plot_fig_path)

        self.model_plot_data_path = os.path.join(self.model_path, 'plots_data')
        if not os.path.exists(self.model_plot_data_path):
            os.mkdir(self.model_plot_data_path)
        
        if IS_HIDDEN_DIST:
            self.hidden_dist_dir = os.path.join(self.model_plot_fig_path, 'HiddenOutDist')
            if not os.path.exists(self.hidden_dist_dir):
                os.makedirs(self.hidden_dist_dir)

##NOTE old code for plot MI plane
    # def plot_MI_plane_1(self, MI_X_T, MI_Y_T):
    #     '''
    #     plot evolution of mutual information for each layer at different eporchs
    #     '''
    #     fig = plt.figure(figsize=(7,7))#, facecolor='#edf0f8')
    #     # f, ax = plt.subplots(1,1)
    #     ax = fig.add_subplot(1,1,1)

    #     # set colormap and font
    #     sm = plt.cm.ScalarMappable(cmap='gnuplot', 
    #                                norm=plt.Normalize(vmin=0, vmax=self._opt.max_epoch))
    #     sm._A = []
    #     csfont = {'fontname':'Times New Roman'}
        
    #     Lepoch = MI_X_T.keys()
    #     for epoch in Lepoch:
    #         ax.plot(MI_X_T[epoch], MI_Y_T[epoch], c=sm.to_rgba(epoch), alpha=0.1, zorder=1)
    #         ax.scatter(MI_X_T[epoch], MI_Y_T[epoch], s=40, facecolor=sm.to_rgba(epoch), zorder=2)

    #     ax.set_title('Information Plane', fontsize = 26, y=1.04, **csfont)
    #     ax.set_xlabel('$\mathcal{I}(X;T)$', fontsize = 22)
    #     ax.set_ylabel('$\mathcal{I}(Y;T)$', fontsize = 22)
    #     # ax.set_xlim(left = 0.)
    #     # ax.set_ylim(bottom = 0.)
    #     ax.set_aspect(1. / ax.get_data_ratio())
    #     ax.set_facecolor('#edf0f8')
    #     ax.grid(color='w', linestyle='-.', linewidth = 1)
    #     ax.tick_params(labelsize = 13)

    #     # cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
    #     fig.colorbar(sm, label='Epoch', fraction=0.0454, pad=0.05)#, cax=cbaxes)

    #     # set dir for mean_std; saving figure
    #     self._save_fig(fig, 'InfoPlan_original')
##

    def plot_MI_plane(self, MI_X_T, MI_Y_T, IS_LAYERWISE_PLOT = True):
        '''
        plot evolution of mutual information for each layer at different eporchs
        MI_X_T & MI_Y_T: dictionary, key -> #epoch, value -> List of len Nlayers 
        '''
        idx = list(MI_X_T.keys())[0] # fix bug when only cal. for only part of the epochs
        Nlayers = len(MI_X_T[idx])
        Lepoch = MI_X_T.keys()

        # set colormap
        sm = plt.cm.ScalarMappable(cmap='gnuplot', 
                                   norm=plt.Normalize(vmin=0, vmax=self._opt.max_epoch))
        sm._A = []
        
        ## plot MI in info plane, containing all layers
        fig_ = plt.figure(figsize=(7,7))
        ax_ = fig_.add_subplot(1,1,1)

        for epoch in Lepoch:
            ax_.plot(MI_X_T[epoch], MI_Y_T[epoch], c=sm.to_rgba(epoch), alpha=0.1, zorder=1)
            ax_.scatter(MI_X_T[epoch], MI_Y_T[epoch], s=40, facecolor=sm.to_rgba(epoch), zorder=2)

        # ax setting
        self._commom_ax_setting_MI_plane(ax_)
        # ax_.set_xlim(left = 0.)
        # ax_.set_ylim(bottom = 0.)
        
        # set color bar
        fig_.colorbar(sm, label='Epoch', fraction=0.0454, pad=0.05)

        # saving figure (single figure containing all layers)
        self._save_fig(fig_, 'InfoPlan_original')

        ## plot MI in info plane, one figure for each layer
        if IS_LAYERWISE_PLOT:

            ax_xrange = ax_.get_xlim()
            ax_yrange = ax_.get_ylim()

            # auto adapted to number of layers
            nrow = int(np.ceil(Nlayers/3))
            fig = plt.figure(figsize=(20,6*nrow), constrained_layout=False)
            gs = GridSpec(nrow, 3, figure=fig, wspace=0.15, hspace=0.25)

            # plotting
            L = -1
            termin = False
            for i in range(3):
                if termin:
                    break
                for j in range(3):
                    L += 1
                    if L >= Nlayers:
                        termin = True
                        break

                    ax = fig.add_subplot(gs[i, j])
                    for epoch in Lepoch:
                        ax.scatter(MI_X_T[epoch][L], MI_Y_T[epoch][L], s=40, facecolor=sm.to_rgba(epoch))

                    # ax setting (note must set_x(y)lim before self._commom_ax_setting_MI_plane)
                    ax.set_xlim(ax_xrange)
                    ax.set_ylim(ax_yrange)
                    self._commom_ax_setting_MI_plane(ax, layer_idx = L)

            fig.subplots_adjust(left = 0.05, bottom=0.05, top=0.95, right=0.9)
            cbaxes = fig.add_axes([0.91, 0.05, 0.03, 0.9]) #rect = l,b,w,h
            fig.colorbar(sm, label='Epoch', cax=cbaxes)

            # saving figure
            self._save_fig(fig, 'InfoPlan')

    def _commom_ax_setting_MI_plane(self, ax, layer_idx = -1):
        if layer_idx == -1:
            ax.set_title('Information Plane'+" ("+self._opt.activation+")", fontsize = 20, y=1.04)
        else:
            ax.set_title('Information Plane (layer'+str(layer_idx+1)+" "+self._opt.activation+")", fontsize = 20)
        ax.set_xlabel('$\mathcal{I}(X;T)$', fontsize = 22)
        ax.set_ylabel('$\mathcal{I}(Y;T)$', fontsize = 22)
        ax.set_aspect(1. / ax.get_data_ratio())
        ax.set_facecolor('#edf0f8')
        ax.grid(color='w', linestyle='-.', linewidth = 1)
        ax.tick_params(labelsize = 13)

## NOTE old plot mean std code for single mean_std figure, 2019-05-24
    # def plot_mean_std(self, Lepoch, mu, sigma):
    #     '''
    #     plot the variation of mean and standard devidation for each layer with respect to epoch

    #     Lepoch    --- array of recorded epochs; of dim (Nepoch,)
    #     mu, sigma --- mean & standard deviation; of dim (Nlayers, feature_dim)
    #     '''
    #     Nlayers = mu.shape[1]

    #     fig = plt.figure(figsize=(9,7))
    #     ax = fig.add_subplot(1,1,1)
    #     legend_mean = []
    #     legend_std  = []
    #     layer_mark = ['layer'+str(i+1) for i in range(Nlayers)]
        
    #     # set color and font
    #     csfont = {'fontname':'Times New Roman'}
    #     colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'darkgreen']
        
    #     for L in range(Nlayers):
    #         legend_mean += ax.plot(Lepoch, mu[:,L], c = colors[L] ,ls='-')
    #         legend_std  += ax.plot(Lepoch, sigma[:,L], c = colors[L], ls='-.')
    
    #     # ax settings
    #     ax.set_title(self._opt.activation)
    #     fig.subplots_adjust(right = 0.86)
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.set_ylim(bottom=1.e-5)
    #     ax.set_xlabel('number of epochs', fontsize=22, **csfont)
    #     ax.set_ylabel('Means and Standard Deviations', fontsize=22, **csfont)
    #     ax.set_facecolor('#edf0f8')
    #     ax.grid(color='w', linestyle='-.', linewidth=1)
    #     ax.tick_params(labelsize=13)
    #     leg_mean = ax.legend(legend_mean, layer_mark,  bbox_to_anchor=[1.15, 1], title='Mean')
    #     leg_std  = ax.legend(legend_std, layer_mark,  bbox_to_anchor=[1.15, 0.6], title='STD')
    #     ax.add_artist(leg_mean)
    #     ax.add_artist(leg_std)

    #     # set dir for mean_std; saving figure
    #     self._save_fig(fig, 'Mean_and_STD')
##

    def plot_mean_std(self, Lepoch, mu, sigma):
        '''
        plot the variation of mean and standard devidation for each layer with respect to epoch

        Lepoch    --- array of recorded epochs; of dim (Nepoch,)
        mu, sigma --- mean & standard deviation; of dim (Nlayers, feature_dim)
        '''

        Nlayers = mu.shape[1]

        fig = plt.figure(figsize=(18,14), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, wspace=0.0, hspace=0.0)

        # intial legend setting
        legend_mean = []
        legend_std  = []
        layer_mark = ['layer'+str(i+1) for i in range(Nlayers)]
        
        # set color
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'darkgreen']

        # plotting
        # 1- mixed mean and std
        ax1 = fig.add_subplot(gs[0, 0])
        for L in range(Nlayers):
            ax1.plot(Lepoch, mu[:,L], c = colors[L] ,ls='-')
            ax1.plot(Lepoch, sigma[:,L], c = colors[L], ls='-.')

        # ax1.set_ylim(bottom=1.e-5)
        self._commom_ax_setting_mean_std(ax1, "Mean and STD", show_xlabel=False)

        # 2- mean
        ax2 = fig.add_subplot(gs[0, 1])
        for L in range(Nlayers):
            legend_mean += ax2.plot(Lepoch, mu[:,L], c = colors[L] ,ls='-')

        ax2.set_ylim(ax1.get_ylim())
        self._commom_ax_setting_mean_std(ax2, "Mean", show_ylabel=False)

        # 3- std
        ax3 = fig.add_subplot(gs[1, 0])
        for L in range(Nlayers):
            legend_std  += ax3.plot(Lepoch, sigma[:,L], c = colors[L], ls='-.')

        ax3.set_ylim(ax1.get_ylim())
        self._commom_ax_setting_mean_std(ax3, "STD")        

        # set legend
        fig.legend(legend_mean, layer_mark, bbox_to_anchor = [0.75, 0.38], 
                   title="Mean", title_fontsize = 17, fontsize = 17)
        fig.legend(legend_std,  layer_mark, bbox_to_anchor=[0.85, 0.38],  
                   title="STD",  title_fontsize = 17, fontsize = 17)

        # set dir for mean_std; saving figure
        self._save_fig(fig, 'Mean_and_STD')

        # # show pic
        # plt.show()
    
    def _commom_ax_setting_mean_std(self, ax, title_name, show_xlabel=True, show_ylabel=True):
        ax.set_title(title_name+" ("+self._opt.activation+")", fontsize=17)
        if show_xlabel:
            ax.set_xlabel('number of epochs', fontsize=19)
        if show_ylabel:
            ax.set_ylabel('Means and Standard Deviations', fontsize=19)

        ax.set_xscale('log')
        # ax.set_yscale('log')

        ax.set_facecolor('#edf0f8')
        ax.grid(color='w', linestyle='-.', linewidth = 1)
        ax.tick_params(labelsize = 13)


    def plot_svd(self, Lepoch, svd):
        '''
        plot for both original and normalized versions
        '''
        self._func_plot_svd(Lepoch, np.array(svd[0]), "_original")
        self._func_plot_svd(Lepoch, np.array(svd[1]), "_normalized")

    def _func_plot_svd(self, Lepoch, weight_svd, nameflag):
        '''
        plot the variation of singular value for the averaged weight of each layer with respect to epoch

        Lepoch    --- array of recorded epochs; of dim (Nepoch,)
        svd       --- list, [[svd_w] [svd_grad]]; [svd_w] = [ [ svd_w_layer_1, ... svd_w_layer_n ]_epoch_1, ... ]
        '''
        Nlayers = len(weight_svd[0])

        nrow = int(np.ceil(Nlayers/3))
        fig = plt.figure(figsize=(23,6*nrow), constrained_layout=True)
        gs = GridSpec(nrow, 3, figure=fig, wspace=0.0, hspace=0.0)

        # set color and font
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'darkgreen']
        cy = cycler('color', colors)
        # csfont = {'fontname':'Times New Roman'}

        # initialize legend settting
        # legend_svd_w = []
        # layer_mark = ['layer'+str(i+1) for i in range(Nlayers)]

        # plotting
        L = -1
        termin = False
        for i in range(3):
            if termin:
                break
            for j in range(3):
                L += 1
                if L >= Nlayers:
                    termin = True
                    break
                svd_val = list(weight_svd[:,L])
                ax = fig.add_subplot(gs[i, j])
                ax.plot(Lepoch, svd_val ,ls='-', marker='o', ms = 4)
    
                # ax settings
                ax.set_prop_cycle(cy)
                ax.set_title('layer'+str(L+1)+' ('+self._opt.activation+')', fontsize=17)
                ax.set_xscale('log')
                ax.set_ylim(bottom = 0.)
                if i == nrow - 1:
                    ax.set_xlabel('number of epochs', fontsize = 19)
                if j == 0:
                    ax.set_ylabel('Singular Values', fontsize = 19)
                ax.set_facecolor('#edf0f8')
                ax.grid(color='w', linestyle='-.', linewidth = 1)
                ax.tick_params(labelsize = 13)
        
        # set legend
        # leg_svd_w = ax.legend(legend_svd_w, layer_mark,  bbox_to_anchor=[1.15, 1], title='svd_w')
        # ax.add_artist(leg_svd_w)

        # set dir for mean_std; saving figure
        self._save_fig(fig, 'SingularValues'+nameflag)

    
    def plot_acc_loss(self, Lepoch, acc_train, acc_test, loss):
        '''
        plot the variation of mean and standard devidation for each layer with respect to epoch

        Lepoch    --- array of recorded epochs; of dim (Nepoch,)
        acc_train --- list of training accuracy; of dim (Nepoch,)
        acc_test  --- list of test accuracy; of dim (Nepoch,)
        loss      --- list of traning loss; of dim (Nepoch,)
        '''
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.plot(Lepoch, acc_train, ls='-', color=color, label='training acc')
        ax1.plot(Lepoch, acc_test, ls='-.', color=color, label='test acc')

        ax1.legend(bbox_to_anchor=[0.95, 0.6])
        ax1.set_xlabel('number of epochs', fontsize = 18)
        ax1.set_ylabel('Training and Testing Accuracy', fontsize = 18)
        # ax1.set_ylim(top = 1.2)
        ax1.set_facecolor('#edf0f8')
        ax1.grid(color='w', linestyle='-.', linewidth = 1)
        ax1.tick_params(labelsize = 13)

        ax2 = ax1.twinx()

        color = 'tab:orange'
        ax2.plot(Lepoch, loss, ls='-', color=color, label='Loss')

        ax2.legend(bbox_to_anchor=[0.95, 0.4])
        ax2.set_ylabel('Loss', fontsize = 18)
        ax2.tick_params(labelsize = 13)

        fig.suptitle("Accuracy and Loss", fontsize=20)
        fig.subplots_adjust(left = 0.14, bottom=0.14, top=0.9, right=0.88)

        # set dir for mean_std; saving figure
        self._save_fig(fig, 'Acc_and_Loss')


    def plot_hidden_dist(self, epoch, layer_activity):
        Nlayers = len(layer_activity)

        fig = plt.figure(figsize=(16,4*Nlayers), constrained_layout=True)
        gs = GridSpec(Nlayers, 1, figure=fig, wspace=0.0, hspace=0.3)

        for i in range(Nlayers):
            data = layer_activity[i].reshape(-1)
            ax = fig.add_subplot(gs[i, 0])
            ax.hist(data, bins = 50)
            ax.set_xlabel('hidden layer outputs value', fontsize = 24)
            ax.set_ylabel('counts', fontsize = 24)
            ax.tick_params(labelsize = 16)
            if i != Nlayers - 1:
                ax.set_xlim(left=self._opt.Vmin, right=self._opt.Vmax)

        fig.suptitle(f'epoch: {str(epoch)} distribution of hidden layer outputs', fontsize=28)
        fig.subplots_adjust(left = 0.1, bottom=0.05, top=0.95, right=0.95)

        fname = os.path.join(self.hidden_dist_dir, str(epoch) + '.png')
        fig.savefig(fname, format='png')
    
    def generate_hidden_dist_gif(self):
        file_names = [fn for fn in os.listdir(self.hidden_dist_dir) if fn.endswith('.png')]
        if len(file_names) == 0:
            raise ValueError('not enough data')
        list.sort(file_names, key=lambda x: int(x.split('.')[0]))
        file_names = [os.path.join(self.hidden_dist_dir, fn) for fn in file_names]
        clip = mpy.ImageSequenceClip(file_names, fps=2)
        filename = os.path.join(self.model_plot_fig_path, "Hidden_Output_Distribution.gif")
        clip.write_gif(filename, fps=2)



    def _save_fig(self, fig, fig_name):
        '''
        we use pdf rather then eps since eps in matplotlib doesn't support transparency
        '''
        # save in model_path
        fig_name_eps = os.path.join(self.model_plot_fig_path, "{}.pdf".format(fig_name))
        fig.savefig(fig_name_eps, format='pdf')

        fig_name_jpg = os.path.join(self.model_plot_fig_path, "{}.jpg".format(fig_name))
        fig.savefig(fig_name_jpg, format='jpeg')

        # save in plot_root
        fig_name_eps = os.path.join(self.plot_path, "{}_{}.pdf".format(fig_name, self.model_name))
        fig.savefig(fig_name_eps, format='pdf')

        fig_name_jpg = os.path.join(self.plot_path, "{}_{}.jpg".format(fig_name, self.model_name))
        fig.savefig(fig_name_jpg, format='jpeg')



    def save_plot_data(self, fname, data):
        '''
        call save PLOT data only because we save to the model_plot_data_path
        '''
        save_path = os.path.join(self.model_plot_data_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)



    def post_plot(self, plot_name):
        if not isinstance(plot_name, list):
            raise TypeError('plot_name must be a list of plot types')
        if plot_name == []:
            raise ValueError('plot list empty')
            
        if 'mean_std' in plot_name:
            epoch_data = self._load_plot_data("recorded_epochs_data.pkl")
            mean_data = self._load_plot_data("mean_data.pkl")
            std_data = self._load_plot_data("std_data.pkl")
            self.plot_mean_std(epoch_data, mean_data, std_data)

        if 'svd' in plot_name:
            epoch_data = self._load_plot_data("recorded_epochs_data.pkl")
            svds_data = self._load_plot_data("svds_data.pkl")
            self.plot_svd(epoch_data, svds_data)

        if 'MI_plane' in plot_name:
            IX_data = self._load_plot_data("IX_dic_data.pkl")
            IY_data = self._load_plot_data("IY_dic_data.pkl")
            self.plot_MI_plane(IX_data, IY_data)

        if 'acc_loss' in plot_name:
            full_epoch_list = self._load_plot_data("full_epoch_list_data.pkl")
            acc_train = self._load_plot_data("acc_train_data.pkl")
            acc_test = self._load_plot_data("acc_test_data.pkl")
            loss = self._load_plot_data("loss_data.pkl")
            self.plot_acc_loss(full_epoch_list, acc_train, acc_test, loss)

        plt.show()

    def _load_plot_data(self, fname):
        data_path = os.path.join(self.model_plot_data_path, fname)
        # print(data_path)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data




def main():
    '''test run
    '''
    # Settings for interactive display of plots
    #NOTE for Mac OS, use pythonw to call polt_utils instead of python
    if sys.platform.startswith('darwin'):
        matplotlib.use("WXAgg")
    elif sys.platform.startswith('win32'):
        matplotlib.use("TKAgg")
    else:
        pass

    # C = type('type_C', (object,), {})
    # opt = C()

    # opt.plot_path = './plots'
    # opt.max_epoch = 100
    # opt.activation = 'tanh'

    # test post plot
    # model_name = 'IBNet_test_plot_acc_loss_tanhx_Time_06_25_15_48'
    # save_root = './results'
    pltfig = PlotFigure()
    pltfig.post_plot(['mean_std'])
    

if __name__ == "__main__":
    main()