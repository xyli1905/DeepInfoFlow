import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

np.random.seed()

# matplotlib.use("TKAgg")
'''NOTE
Seem for conda env, one must create a file `matplotlibrc` at the directory `~/.matplotlib`,
and add the following content in this file:
                                            backend : TKAgg
see for reference: https://github.com/matplotlib/matplotlib/issues/13414
'''

class PlotFigure:
    def __init__(self, opt, model_name):
        self.name = 'Plot_Utils'
        self._opt = opt

        # NOTE we save figures in two places: the plot_root and results/model_path
        # check existence of plot_root
        self.plot_dir = opt.plot_dir
        if not os.path.exists(opt.plot_dir):
            os.mkdir(opt.plot_dir)

        self.model_name = model_name
        tmp_dir = os.path.join('./results', model_name)
        self.model_path = os.path.join(tmp_dir, 'plots')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # # set saving directory for present model
        # self.experiment_dir = os.path.join(opt.plot_dir, opt.experiment_name)
        # if not os.path.exists(self.experiment_dir):
        #     os.mkdir(self.experiment_dir)

        # # set timestamp to distinguish same model at differet training
        # timestamp = datetime.datetime.today().strftime('%m_%d_%H_%M')
        # self.timestamp_dir = os.path.join(self.experiment_dir, timestamp)
        # if not os.path.exists(self.timestamp_dir):
        #     os.mkdir(self.timestamp_dir)

    def plot_MI_plane(self, MI_X_T, MI_Y_T):
        '''
        plot evolution of mutual information for each layer at different eporchs
        '''
        fig = plt.figure(figsize=(7,7))#, facecolor='#edf0f8')
        # f, ax = plt.subplots(1,1)
        ax = fig.add_subplot(1,1,1)

        # set colormap and font
        sm = plt.cm.ScalarMappable(cmap='gnuplot', 
                                   norm=plt.Normalize(vmin=0, vmax=self._opt.max_epoch))
        sm._A = []
        csfont = {'fontname':'Times New Roman'}
        
        Lepoch = MI_X_T.keys()
        for epoch in Lepoch:
            ax.plot(MI_X_T[epoch], MI_Y_T[epoch], c=sm.to_rgba(epoch), alpha=0.1, zorder=1)
            ax.scatter(MI_X_T[epoch], MI_Y_T[epoch], s=40, facecolor=sm.to_rgba(epoch), zorder=2)

        ax.set_title('Information Plane', fontsize = 26, y=1.04, **csfont)
        ax.set_xlabel('$\mathcal{I}(X;T)$', fontsize=22)
        ax.set_ylabel('$\mathcal{I}(Y;T)$', fontsize=22)
        # ax.set_xlim(0,8)
        # ax.set_ylim(0,1)
        ax.set_aspect(1. / ax.get_data_ratio())
        ax.set_facecolor('#edf0f8')
        ax.grid(color='w', linestyle='-.', linewidth=1)
        ax.tick_params(labelsize=13)

        # cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
        fig.colorbar(sm, label='Epoch', fraction=0.0454, pad=0.05)#, cax=cbaxes)

        # set dir for mean_std; saving figure
        self._save_fig(fig, 'InfoPlan')


    def plot_mean_std(self, Lepoch, mu, sigma):
        '''
        plot the variation of mean and standard devidation for each layer with respect to epoch

        Lepoch    --- array of recorded epochs; of dim (Nepoch,)
        mu, sigma --- mean & standard deviation; of dim (Nlayers, feature_dim)
        '''

        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(1,1,1)
        legend_mean = []
        legend_std  = []
        layer_mark = ['layer'+str(i+1) for i in range(mu.shape[1])]
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'darkgreen']

        # set color and font
        csfont = {'fontname':'Times New Roman'}

        Nlayers = mu.shape[1]
        for L in range(Nlayers):
            legend_mean += ax.plot(Lepoch, mu[:,L], c = colors[L] ,ls='-')
            legend_std  += ax.plot(Lepoch, sigma[:,L], c = colors[L], ls='-.')
    
        # ax settings
        fig.subplots_adjust(right = 0.86)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1.e-5)
        ax.set_xlabel('number of epochs', fontsize=22, **csfont)
        ax.set_ylabel('Means and Standard Deviations', fontsize=22, **csfont)
        ax.set_facecolor('#edf0f8')
        ax.grid(color='w', linestyle='-.', linewidth=1)
        ax.tick_params(labelsize=13)
        leg_mean = ax.legend(legend_mean, layer_mark,  bbox_to_anchor=[1.15, 1], title='Mean')
        leg_std  = ax.legend(legend_std, layer_mark,  bbox_to_anchor=[1.15, 0.6], title='STD')
        ax.add_artist(leg_mean)
        ax.add_artist(leg_std)

        # set dir for mean_std; saving figure
        self._save_fig(fig, 'Mean_and_STD')


    def plot_other(self):
        pass


    def _save_fig(self, fig, fig_name):
        # save in model_path
        fig_name_eps = os.path.join(self.model_path, "{}.eps".format(fig_name))
        fig.savefig(fig_name_eps, format='eps')

        fig_name_jpg = os.path.join(self.model_path, "{}.jpg".format(fig_name))
        fig.savefig(fig_name_jpg, format='jpeg')

        # save in plot_root
        fig_name_eps = os.path.join(self.plot_dir, "{}_{}.eps".format(fig_name, self.model_name))
        fig.savefig(fig_name_eps, format='eps')

        fig_name_jpg = os.path.join(self.plot_dir, "{}_{}.jpg".format(fig_name, self.model_name))
        fig.savefig(fig_name_jpg, format='jpeg')




def main():
    '''test run
    '''
    # test data for plot_MI_plane
    x1 = np.array([0.51842304, 0.92556737, 0.36004445, 0.11063085, 0.89165   ])
    y1 = np.array([0.63147293, 0.59704809, 0.67011044, 0.01976542, 0.95609   ])
    x2 = np.array([0.52649129, 0.45103952, 0.63225806, 0.0176416,  0.94888   ])
    y2 = np.array([0.63147293, 0.59704809, 0.67011044, 0.01976542, 0.95609   ])

    # test data for plot_MI_plane
    Lepoch = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900])
    mu = np.random.rand(1, Lepoch.shape[0])
    sigma = np.random.rand(1, Lepoch.shape[0])


    C = type('type_C', (object,), {})
    opt = C()

    opt.plot_dir = './plots'
    opt.experiment_name = 'testdrawing'
    # opt.timestamp = '19050310'

    # pltfig = PlotFigure(opt)

    # pltfig.plot_MI_plane(x1,y2,x2,y2)

    # pltfig.plot_mean_std(Lepoch, mu, sigma)
    


if __name__ == "__main__":
    main()