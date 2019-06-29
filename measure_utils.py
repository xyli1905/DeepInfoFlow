import torch
import numpy as np
from log_barrier import LogBarrier


# --------------------------------------------------------------------------------------- #
# collection of methods for kde;
# --------------------------------------------------------------------------------------- #
def Kget_dists(X):
    """Torch code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = torch.unsqueeze(torch.sum(torch.pow(X, 2), dim=1), dim=1)
    dists = x2 + torch.transpose(x2, 0, 1) - 2*torch.matmul(X, torch.transpose(X, 0,1))
    return dists

def get_shape(x):
    '''
    assuming x is torch tensor
    '''
    dims = x.size()[1]
    N    = x.size()[0]
    return dims, N

def entropy_estimator_kl(x, var):
    '''
    KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
        see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
        and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    '''
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2.0*var)
    normconst = (dims/2.0)*np.log(2.0*np.pi*var)
    lprobs = torch.logsumexp(-dists2, dim=1) - np.log(N) - normconst
    h = -torch.mean(lprobs)
    return dims/2.0 + h

def entropy_estimator_bd(x, var):
    '''
    Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
        see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    '''
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    '''
    Return entropy of a multivariate Gaussian, in nats
    '''
    dims = output.shape[1] # np ndarray?
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


# --------------------------------------------------------------------------------------- #
# measre based on the Empirical estimation of Variational form for K-L divergence (EVKL)
# collection of methods for EVKL;
# --------------------------------------------------------------------------------------- #
def MI_estimator(samples_x, samples_y):
    '''
    Empirically estimate the KL div between dist(x) and dist(y)
    where
        samples_x: drawn from either P(X,T) or P(Y,T)
        samples_y: drawn from either P(X)P(T) or P(Y)P(T)
    NOTE
        Dkl(P(X,T)||P(X)P(T)) = I(X;T) and
        Dkl(P(Y,T)||P(Y)P(T)) = I(Y;T)
        so we are in fact estimating the MI between X, T and Y, T
    '''
    Q, c = prepare_Q_c(samples_x, samples_y)

    l = LogBarrier()
    alpha = l.compute_alpha(Q, c)

    Dkl_hat = estimate_KL_div(alpha)

    return Dkl_hat

def prepare_Q_c(x, y):
    '''
    Compute:
        first, Kyy := K(yi,yj) and
                Kxy := K(xi, yj);
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

    # parameters, for test
    lambdaN = 1./Nsamples
    sigma = 1.0

    # compute Kxy
    var = np.zeros((Nsamples, Nsamples))
    for i in range(Nsamples):
        for j in range(Nsamples):
            delta = x[i,:] - y[j,:]
            mu2 = np.dot(delta, delta)
            var[i,j] = -1.0 * mu2 / sigma
    Kxy = np.exp(var)

    # compute Kyy
    var = np.zeros((Nsamples, Nsamples))
    for i in range(Nsamples):
        for j in range(i, Nsamples):
            delta = y[i,:] - y[j,:]
            mu2 = np.dot(delta, delta)
            var[i,j] = -1.0 * mu2 / sigma
            var[j,i] = var[i,j]
    Kyy = np.exp(var)

    Q = Kyy / lambdaN
    c = np.transpose(Kxy) @ np.ones((Nsamples, 1)) / (-1.0 * lambdaN * Nsamples)

    return Q, c

def estimate_KL_div(alpha):
    '''
    based on the formular of
        Estimating divergence functional and the likelihood ratio by penalized
        convex risk minimization, XuanLong Nguyen, et al (2010)
    '''
    N = alpha.shape[0]
    # print(alpha)
    return -np.log(N) - np.sum(np.log(alpha))/N



if __name__ == '__main__':
    dim = 1000

    x = np.random.rand(dim, 16)
    y = np.random.rand(dim, 16)

    Dkl_test = MI_estimator(x, y)

    print(Dkl_test)