from cvxopt import solvers, matrix, spdiag, log
import numpy as np

def acent(Q, c, G, h):
    n, n = G.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        # f = -sum(log(x))/ n + 0.5 * x.T * Q * x + c.T * x
        f =  0.5 * x.T * Q * x + c.T * x
        # Df = -(x**-1).T + x.T * Q + c.T
        Df = x.T * Q + c.T
        if z is None: return f, Df
        H = spdiag(z[0] * x**-2)
        return f, Df, H
    return solvers.cp(F, G=G, h=h)['x']


if __name__ == '__main__':
    dim = 100
    Q_n = np.random.rand(dim, dim)
    Q_n = Q_n.T @ Q_n
    c_n = np.random.rand(dim, 1)

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
    sol = np.array(solvers.qp(Q, c, G = G, h= h)['x'])
    
    test1 = 0.5 * res.T @ Q_n @ res + c.T @ res
    test2 = 0.5 * sol.T @ Q_n @ sol + c.T @ sol
    print(test1, test2)


# from cvxopt import matrix, log, div, spdiag, solvers

# def F(x = None, z = None):
#      if x is None:  return 0, matrix(0.0, (3,1))
#      if max(abs(x)) >= 1.0:  return None
#      u = 1 - x**2
#      val = -sum(log(u))
#      Df = div(2*x, u).T
#      if z is None:  return val, Df
#      H = spdiag(2 * z[0] * div(1 + u**2, u**2))
#      return val, Df, H

# G = matrix([ [0., -1.,  0.,  0., -21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
#              [0.,  0., -1.,  0.,   0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
#              [0.,  0.,  0., -1.,  -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.] ])
# h = matrix([1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.])
# dims = {'l': 0, 'q': [4], 's':  [3]}
# sol = solvers.cp(F, G, h, dims)
# print(sol['x'])
# # [ 4.11e-01]
# # [ 5.59e-01]
# # [-7.20e-01]
