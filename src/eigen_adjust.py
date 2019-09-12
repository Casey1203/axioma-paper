import numpy as np
import pandas as pd


def sort_eigen_pair(val, vec, sort='descend'):
    # 对特征向量按照特征值从大到小排序
    if sort == 'descend':
        idx = val.argsort()[::-1]
    else:
        idx = val.argsort()[::1]
    val = val[idx]
    vec = vec[:, idx]
    return val, vec


def eigen_adjustment(sample_covariance, M, T, a=1.4):
    val, vec = np.linalg.eig(sample_covariance)
    val, vec = sort_eigen_pair(val, vec)
    U_0 = np.mat(vec) # 特征矩阵
    V_0 = np.mat(sample_covariance)
    D_0 = U_0.T * V_0 * U_0

    # generate M set of simulated eigenfactor return b
    variance_list = np.diag(D_0)
    D_ratio = {}
    for m in range(M):
        b_m = {}
        for i in range(len(variance_list)):
            b_m[i] = np.random.normal(0., np.sqrt(variance_list[i]), T)
        b_m = pd.DataFrame(b_m).T
        f_m = U_0 * np.mat(b_m)
        # V_m = (f_m * f_m.T) / (len(variance_list) - 1)
        V_m = np.cov(f_m, ddof=1)
        val_m, vec_m = np.linalg.eig(V_m)
        val_m, vec_m = sort_eigen_pair(val_m, vec_m)
        U_m = np.mat(vec_m)
        D_m = U_m.T * V_m * U_m
        diag_D_m = np.diag(D_m)
        D_tilde_m = U_m.T * V_0 * U_m
        # D_ratio['m_%s' % m] = np.sqrt(np.diag(D_tilde_m) / np.diag(D_m))
        D_ratio['m_%s' % m] = np.nan_to_num(np.sqrt(np.diag(D_tilde_m) / np.diag(D_m)), 0.)
    D_ratio = pd.DataFrame(D_ratio).T

    vol_bias = D_ratio.mean()

    gamma = a * (vol_bias - 1) + 1

    D_tilde_0 = D_0 * np.mat(np.diag(gamma) ** 2)

    V_tilde_0 = U_0 * D_tilde_0 * U_0.T

    # adjust_covariance = pd.DataFrame(index=sample_covariance.index, columns=sample_covariance.columns, data=V_tilde_0)

    return V_tilde_0

if __name__ == '__main__':
    eigen_adjustment(
        [[1., 0.5],
         [0.5, 1.]
         ], 1, 5
    )