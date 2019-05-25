import torch
import numpy as np


class kde:
    def __init__(self):
        pass

    def Kget_dists(self, X):
        """Torch code to compute the pairwise distance matrix for a set of
        vectors specifie by the matrix X.
        """
        x2 = torch.unsqueeze(torch.sum(torch.pow(X, 2), dim=1), dim=1)
        dists = x2 + torch.transpose(x2, 0, 1) - 2*torch.matmul(X, torch.transpose(X, 0,1))
        # print('________________________________________________________________________________')
        # print(dists[:10, :10])
        # print('________________________________________________________________________________')
        return dists


    def get_shape(self, x):
        '''
        assuming x is torch tensor
        '''
        dims = x.size()[1]
        N    = x.size()[0]
        return dims, N
    
    def entropy_estimator_kl(self, x, var):
        '''
        KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
         see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
         and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
        '''
        dims, N = self.get_shape(x)
        dists = self.Kget_dists(x)
        dists2 = dists / (2.0*var)
        normconst = (dims/2.0)*np.log(2.0*np.pi*var)
        lprobs = torch.logsumexp(-dists2, dim=1) - np.log(N) - normconst
        h = -torch.mean(lprobs)
        return dims/2.0 + h

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


# measre based on the variational empirical estimation of K-L divergence
class VEKL:
    def __init__(self):
        pass

    def MI_estimator(self):
        pass

    def _prepare_Kyy_Kxy(self, x, y):
        '''
        Compute: 
            first, Kyy := K(yi,yj) and Kxy := K(xi, yj);
            then Q = 1/lambdaN * Kyy, and
                 c = - 1/(n*lambdaN) * Kxy^T @ ones
        where
            K(x, y) = exp(-||x - y||^2 / sigma),
            hyper params: lambdaN, sigma
            (ones are n*1 matrix with all element being 1.)

        return Q and c for method _QP_log_barrier()
        '''
        # get number of samples
        Nsamples = x.shape[0]

        # parameters, for test presently
        lambdaN = 1./Nsamples
        sigma = 1.0

        # compute Kxy
        var = np.zeros(Nsamples, Nsamples)
        for i in range(Nsamples):
            for j in range(Nsamples):
                delta = x[i,:] - y[j,:]
                mu2 = np.dot(delta, delta)
                var[i,j] = -1.0 * mu2 / sigma
        Kxy = np.exp(var)
        
        # compute Kyy
        var = np.zeros(Nsamples, Nsamples)
        for i in range(Nsamples):
            for j in range(i, Nsamples):
                delta = y[i,:] - y[j,:]
                mu2 = np.dot(delta, delta)
                var[i,j] = -1.0 * mu2 / sigma
                var[j,i] = var[i,j]
        Kyy = np.exp(var)

        Q = Kyy/lambdaN
        c = np.transpose(Kxy) @ np.ones(Nsamples, 1) / (-1.0 * lambdaN * Nsamples)

        return Q, c 
        

    def _QP_log_barrier(self):
        pass