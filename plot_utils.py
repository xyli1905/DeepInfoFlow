import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, opt):
        self.name = 'Plot_Utils'
        self._opt = opt

        # check root saving directory
        if not os.path.exists(opt.plot_dir):
            os.mkdir(opt.plot_dir)

        # set saving directory for present model
        self.model_dir = os.path.join(opt.plot_dir, opt.model_name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # set timestamp to distinguish same model at differet training
        self.timestamp_dir = os.path.join(self.model_dir, opt.timestamp)
        if not os.path.exists(self.timestamp_dir):
            os.mkdir(self.timestamp_dir)

    def plot_MI_plane(self, x1,y1,x2,y2):
        '''
        plot evolution of mutual information for each layer at different eporchs
        '''
        fig = plt.figure(figsize=(7,7))#, facecolor='#edf0f8')
        # f, ax = plt.subplots(1,1)
        ax = fig.add_subplot(1,1,1)

        # set colormap and font
        sm = plt.cm.ScalarMappable(cmap='gnuplot', 
                                   norm=plt.Normalize(vmin=0, vmax=1000))
        sm._A = []
        csfont = {'fontname':'Times New Roman'}

        ##-will be loop over epoch
        ax.plot(x1,y1, c=sm.to_rgba(600), alpha=0.1, zorder=1)
        ax.scatter(x1,y1, s=60, facecolor=sm.to_rgba(600), zorder=2)

        ax.plot(x2,y2, c=sm.to_rgba(200), alpha=0.1, zorder=1)
        ax.scatter(x2,y2, s=60, facecolor=sm.to_rgba(200), zorder=2)
        ##-

        ax.set_title('Information Plane', fontsize = 26, y=1.04, **csfont)
        ax.set_xlabel('$\mathcal{I}(X;T)$', fontsize=22)
        ax.set_ylabel('$\mathcal{I}(Y;T)$', fontsize=22)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('#edf0f8')
        ax.grid(color='w', linestyle='-.', linewidth=1)
        ax.tick_params(labelsize=13)

        # cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
        fig.colorbar(sm, label='Epoch', fraction=0.0454, pad=0.05)#, cax=cbaxes)

        # set dir for InfoPlan
        fig_dir = os.path.join(self.timestamp_dir, 'InfoPlan')
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        fig_name = os.path.join(fig_dir, "test.eps")
        fig.savefig(fig_name, format='eps')


    def plot_mean_std(self, Lepoch, mu, sigma):
        '''
        plot the variation of mean and standard devidation for each layer with respect to epoch

        Lepoch    --- array of recorded epochs; of dim (Nepoch,)
        mu, sigma --- mean & standard deviation; of dim (Nlayers, feature_dim)
        '''

        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(1,1,1)

        # set color and font
        csfont = {'fontname':'Times New Roman'}

        Nlayers = mu.shape[0]
        for L in range(Nlayers):
            ax.plot(Lepoch, mu[L,:], ls='-')
            ax.plot(Lepoch, sigma[L,:], ls='-.')
            
        # ax settings
        ax.set_xscale('log')
        ax.set_xlabel('number of epochs', fontsize=22, **csfont)
        ax.set_ylabel('Means and Standard Deviations', fontsize=22, **csfont)
        # ax.set_facecolor('#edf0f8')
        # ax.grid(color='w', linestyle='-.', linewidth=1)
        ax.tick_params(labelsize=13)

        # set dir for mean_std; saving figure
        fig_dir = os.path.join(self.timestamp_dir, 'Mean_and_STD')
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        fig_name = os.path.join(fig_dir, "test.eps")
        fig.savefig(fig_name, format='eps')


    def plot_other(self):
        pass



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
    opt.model_name = 'testdrawing'
    opt.timestamp = '19050310'

    pltfig = PlotFigure(opt)

    pltfig.plot_MI_plane(x1,y2,x2,y2)

    pltfig.plot_mean_std(Lepoch, mu, sigma)
    


if __name__ == "__main__":
    main()