import numpy as np


def calc_volatility_by_V(V, h):
    V_mat = np.mat(V.values)
    h = np.mat(h).T
    var = h.T * V_mat * h
    return var[0, 0]
