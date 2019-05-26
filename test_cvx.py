from cvxopt import solvers, matrix, spdiag, log
import numpy as np

def acent(Q, c, G, h):
    n, n = G.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f = -sum(log(x))/n + 0.5 * x.T * Q * x + c.T * x
        # f =  0.5 * x.T * Q * x + c.T * x # test for qp
        Df = -(x**-1).T/n + x.T * Q + c.T
        # Df = x.T * Q + c.T # test for qp
        if z is None: return f, Df
        H = spdiag(z[0] * x**-2) + z[0] * Q
        # H = z[0] * Q  # test for qp
        return f, Df, H
    return solvers.cp(F, G=G, h=h)['x']


if __name__ == '__main__':
    dim = 100

    x = np.random.rand(dim, 16)
    y = np.random.rand(dim, 16)

    # compute Kyy
    var = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            delta = x[i,:] - y[j,:]
            mu2 = np.dot(delta, delta)
            var[i,j] = -1.0 * mu2 # sigma == 1
    Kxy = np.exp(var)
    c_n = - np.transpose(Kxy) @ np.ones((dim, 1))
    # print(c_n.shape)
    
    # compute Kyy
    var = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            delta = y[i,:] - y[j,:]
            mu2 = np.dot(delta, delta)
            var[i,j] = -1.0 * mu2 
            var[j,i] = var[i,j]
    Kyy = np.exp(var)
    Q_n = dim * Kyy

    g_n = np.ones(dim) * -1.0
    G_n = np.diag(g_n)
    h_n = np.ones((dim, 1)) * -1.e-10

    # print(g_n)

    Q = matrix(Q_n)
    c = matrix(c_n)
    G = matrix(G_n)
    h = matrix(h_n)

    # print(G_n)

    res = np.array(acent(Q, c, G, h))
    # sol = np.array(solvers.qp(Q, c, G = G, h= h)['x'])
    
    # print(np.linalg.norm(res - sol))

    test1 = 0.5 * res.T @ Q_n @ res + c.T @ res - np.sum(np.log(res)) / dim
    # test2 = 0.5 * sol.T @ Q_n @ sol + c.T @ sol
    # print(test1, test2)
    print(test1)