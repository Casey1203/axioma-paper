import numpy as np
import pandas as pd

def calc_volatility_by_V(V, h):
    V_mat = np.mat(V)
    h = np.mat(h).T
    var = h.T * V_mat * h
    return var[0, 0]


def compute_risk_adjusted_return(r_t, sigma_t): # small b
    if sigma_t == 0.:
        return 0.
    else:
        return r_t / sigma_t

def compute_bias_stats(b_series):
    return b_series.std(ddof=1)

class BiasStats:
    def __init__(self,
                 portfolio_weight_dict,
                 equity_return_dict,
                 covariance_dict,
                 q
                 ):
        self.portfolio_weight_dict = portfolio_weight_dict # key为时间
        self.equity_return_dict = equity_return_dict # key为时间
        self.covariance_dict = covariance_dict # key为时间
        self.q = q

    def compute_ptf_bias_stats(self):
        # ptf_bias_stats_series = pd.Series(index=self.portfolio_weight_dict.keys())
        b_dict = {}  # date as end period
        for t, portfolio_weight in self.portfolio_weight_dict.items():
            # r_ptf_list = pd.Series(index=self.window_range[1:])
            equity_return = self.equity_return_dict[t+1]
            cov = self.covariance_dict[t]
            R_t = portfolio_weight.dot(equity_return)
            total_risk = np.sqrt(calc_volatility_by_V(cov, portfolio_weight))
            b = compute_risk_adjusted_return(R_t, total_risk)
            b_dict[t+1] = b
        b_series = pd.Series(b_dict)
        B_ptf = compute_bias_stats(b_series)
        return B_ptf

    def compute_avg_bias_stats(self, ptf_bias_stats_series):
        return ptf_bias_stats_series.mean()

    def compute_percentile_value(self, series, percentile):
        return series.quantile(q=percentile)

    def compute_mrad(self, ptf_bias_stats_series):
        return abs(ptf_bias_stats_series.subtract(1.)).mean()

    def execute(self):
        ptf_bias_stats_series = self.compute_ptf_bias_stats()
        mean_bias_stats = self.compute_avg_bias_stats(ptf_bias_stats_series)
        percentile_5 = self.compute_percentile_value(ptf_bias_stats_series, 0.05)
        percentile_95 = self.compute_percentile_value(ptf_bias_stats_series, 0.95)
        mrad = self.compute_mrad(ptf_bias_stats_series)
        return pd.Series({
            'mean_bias_stats': mean_bias_stats,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'mrad': mrad
        })
