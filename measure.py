import torch
import numpy as np


class kde:
    def __init__(self):
        self.data = 1111

    def Kget_dists(self, X):
        """Torch code to compute the pairwise distance matrix for a set of
        vectors specifie by the matrix X.
        """
        x2 = torch.unsqueeze(torch.sum(torch.pow(X, 2), dim=1), dim=1)
        dists = x2 + torch.transpose(x2, 0, 1) - 2*torch.matmul(X, torch.transpose(X, 0,1))
        return dists


    def get_shape(self, x):
        '''
        assuming x is torch tensor
        '''
        dims = x.size()[1].double()
        N    = x.size()[0].double()
        return dims, N
    
    def entropy_estimator_kl(self, x, var):
        '''
        KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
         see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
         and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
        '''
        dims, N = self.get_shape(x)
        dists = self.Kget_dists(x)
        dists2 = dists / (2*var)
        normconst = (dims/2.0)*torch.log(2*np.pi*var)
        lprobs = torch.logsumexp(-dists2, dim=1) - torch.log(N) - normconst
        h = -torch.mean(lprobs)
        return dims/2 + h

    def entropy_estimator_bd(self, x, var):
        '''
        Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
         see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
        '''
        dims, N = self.get_shape(x)
        val = self.entropy_estimator_kl(x,4*var)
        return val + np.log(0.25)*dims/2

    def kde_condentropy(self, output, var):
        '''
        Return entropy of a multivariate Gaussian, in nats
        '''
        dims = output.shape[1] # np ndarray?
        return (dims/2.0)*(np.log(2*np.pi*var) + 1)


class NewMeasure:
    def __init__(self):
        pass