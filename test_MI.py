import numpy as np
import measure_utils as measure

np.random.seed()

class build_dataset:
    def __init__(self, dsize):
        self._dsize = dsize
        self._X = np.array([[1.], [2.], [3.]])
        self._Y = np.array([[40.], [50.], [60.]])
        self._XY = np.array([[1., 40.],
                             [2., 50.],
                             [3., 60.]])
        self._XX = np.array([[1., 1.],
                             [2., 2.],
                             [3., 3.]])

    def get_samples(self):
        idx_X = np.random.choice(3, self._dsize, replace=True)
        idx_Y = np.random.choice(3, self._dsize, replace=True)
        idx_XY = np.random.choice(3, self._dsize, replace=True)
        return self._X[idx_X], self._Y[idx_Y], self._XY[idx_XY]
        # return self._X[idx_X], self._X[idx_Y], self._XX[idx_XY]


class EVKL:
    def __init__(self, X, Y, XY):
        self._XY = XY
        self._X_Y = np.concatenate((X, Y), axis=1)
    
    def evalMI(self):
        return measure.MI_estimator(self._XY, self._X_Y)

if __name__ == "__main__":
    dsize = 500
    test_D = build_dataset(dsize)
    X, Y, XY = test_D.get_samples()
    # X1, X2, XX = test_D.get_samples()
    # if dsize <= 10:
    #     for i in range(X.shape[0]):
    #         print(f"{X[i]} {Y[i]} - {XY[i]}")

    evkl = EVKL(X, Y, XY)
    # evkl = EVKL(X1, X2, XX)
    print(evkl.evalMI(), np.log(3))