import cvxpy as cvx
import numpy as np
import pandas as pd
from src.solver import *
from sklearn.metrics.pairwise import cosine_similarity

class RobustMaxReturnSolver(MaxReturnSolver):
    def __init__(self, param):
        super(RobustMaxReturnSolver, self).__init__(param)
        self.param = param.copy()
        self.estimated_alpha_cov = param['estimated_alpha_cov']
        self.kappa = param['kappa']

    def solve_without_round(self):
        solve_result = MaxReturnSolver(self.param).solve_without_round()
        w_tilde = solve_result['expected_weight']
        adjust_term = np.sqrt(self.kappa ** 2 / (w_tilde-self.benchmark_weight_series).T.dot(self.estimated_alpha_cov).dot(w_tilde-self.benchmark_weight_series))
        adjust_alpha = self.alpha_series - adjust_term * self.estimated_alpha_cov.dot(w_tilde-self.benchmark_weight_series)
        self.param['alpha_series'] = adjust_alpha
        # print(
        #     cosine_similarity(adjust_alpha.values.reshape(1, -1), self.param['alpha_true'].values.reshape(1, -1)))
        robust_solve_result = MaxReturnSolver(self.param).solve_without_round()
        return robust_solve_result

# class RobustMaxReturnSolver(MaxReturnSolver):
#     def __init__(self, param):
#         super(RobustMaxReturnSolver, self).__init__(param)
#         self.param = param
#         self.estimated_alpha_cov = param['estimated_alpha_cov']
#         self.kappa = param['kappa']
#
#     def solve_without_round(self):
#         w_c = cvx.Variable(self.alpha_series.size, nonneg=True)
#         t = cvx.Variable(1, nonneg=True)
#         norm = cvx.norm(cvx.matmul(np.linalg.cholesky(self.estimated_alpha_cov), w_c), 2)
#         # 目标函数
#         ret = np.array(self.alpha_series.values)
#         objective = cvx.Maximize(cvx.sum(ret.T * w_c) - self.kappa * norm)
#         # 股票权重 + 现金权重 = 1
#         sum_cons = cvx.sum(w_c) == 1
#         # 风险约束
#         risk_cons = cvx.quad_form((w_c - self.benchmark_weight_series), self.asset_ret_cov) <= self.target_risk ** 2
#         asset_lower_bounds_cons = w_c >= self.asset_lower_boundary_series.values
#         asset_upper_bounds_cons = w_c <= self.asset_upper_boundary_series.values
#         # estimated error constraint
#         # estimated_error_cons = cvx.SOC(t, np.nan_to_num(np.sqrt(self.estimated_alpha_cov), 0.)@w_c)
#         # estimated_error_cons = cvx.square(t) >= cvx.quad_form(w_c, self.estimated_alpha_cov)
#
#         constraints = [sum_cons, risk_cons]
#         if asset_lower_bounds_cons is not None:
#             constraints += [asset_lower_bounds_cons]
#         if asset_upper_bounds_cons is not None:
#             constraints += [asset_upper_bounds_cons]
#
#         for el in constraints:
#             assert (el.is_dcp())
#
#         prob = cvx.Problem(objective, constraints)
#         prob.solve(solver=self.solver, verbose=False)
#
#         # 返回结果
#         after_trade_value = self.market_value * pd.Series(index=self.alpha_series.index, data=w_c.value)
#         after_trade_volume = np.round(after_trade_value / self.current_price)
#         after_trade_volume.fillna(0, inplace=True)
#         after_trade_ratio = after_trade_value / after_trade_value.sum()
#         exante_risk = np.sqrt(after_trade_ratio.T.dot(self.asset_ret_cov).dot(after_trade_ratio))
#         exante_return = after_trade_ratio.T.dot(ret)
#         exante_sr = exante_return / exante_risk
#         solve_result = {
#             "expected_weight": after_trade_ratio,
#             "expected_holding": after_trade_value,
#             "expected_volume": after_trade_volume,
#             "exante_return": exante_return,
#             "exante_risk": exante_risk,
#             "exante_sr": exante_sr,
#             "opt_status": prob.status
#         }
#         return solve_result


# class RobustMaxReturnSolver(MaxReturnSolver):
#     def __init__(self, param):
#         super(RobustMaxReturnSolver, self).__init__(param)
#         self.param = param
#         self.estimated_alpha_cov = param['estimated_alpha_cov']
#         self.kappa = param['kappa']
#
#     def solve_without_round(self):
#         w_c = cvx.Variable(self.alpha_series.size, nonneg=True)
#         eta = cvx.Variable(1, nonneg=True)
#         # 目标函数
#         objective = cvx.Minimize(self.kappa * cvx.quad_form(w_c, self.estimated_alpha_cov))
#         # objective = cvx.Maximize(cvx.sum(ret.T * w_c) - self.kappa * cvx.quad_form(w_c, self.estimated_alpha_cov))
#         # 股票权重 + 现金权重 = 1
#         sum_cons = cvx.sum(w_c) == 1
#         # unit alpha约束
#         ret = np.array(self.alpha_series.values)
#         unit_alpha_cons = ret.T * w_c == eta
#
#         # 风险约束
#         risk_cons = cvx.quad_form((w_c - self.benchmark_weight_series), self.asset_ret_cov) <= self.target_risk ** 2
#         asset_lower_bounds_cons = w_c >= self.asset_lower_boundary_series.values
#         asset_upper_bounds_cons = w_c <= self.asset_upper_boundary_series.values
#         # estimated error constraint
#         # estimated_error_cons = cvx.square(t) >= cvx.quad_form(w_c, self.estimated_alpha_cov)
#
#         constraints = [sum_cons, risk_cons, unit_alpha_cons]
#         if asset_lower_bounds_cons is not None:
#             constraints += [asset_lower_bounds_cons]
#         if asset_upper_bounds_cons is not None:
#             constraints += [asset_upper_bounds_cons]
#
#         for el in constraints:
#             assert (el.is_dcp())
#
#         prob = cvx.Problem(objective, constraints)
#         prob.solve(solver=self.solver, verbose=False)
#
#         # 返回结果
#         after_trade_value = self.market_value * pd.Series(index=self.alpha_series.index, data=w_c.value)
#         after_trade_volume = np.round(after_trade_value / self.current_price)
#         after_trade_volume.fillna(0, inplace=True)
#         after_trade_ratio = after_trade_value / after_trade_value.sum()
#         exante_risk = np.sqrt(after_trade_ratio.T.dot(self.asset_ret_cov).dot(after_trade_ratio))
#         exante_return = after_trade_ratio.T.dot(ret)
#         exante_sr = exante_return / exante_risk
#         solve_result = {
#             "expected_weight": after_trade_ratio,
#             "expected_holding": after_trade_value,
#             "expected_volume": after_trade_volume,
#             "exante_return": exante_return,
#             "exante_risk": exante_risk,
#             "exante_sr": exante_sr,
#             "opt_status": prob.status
#         }
#         return solve_result

class RobustMinRiskSolver(MinRiskSolver):
    def __init__(self, param):
        super(RobustMinRiskSolver, self).__init__(param)
        self.param = param.copy()
        self.estimated_alpha_cov = param['estimated_alpha_cov']
        self.kappa = param['kappa']
        self.z = 0

    def solve_without_round(self):
        solve_result = MinRiskSolver(self.param).solve_without_round()
        w_tilde = solve_result['expected_weight']
        # print('w_tilde', w_tilde)
        adjust_term = np.sqrt(self.kappa ** 2 / (w_tilde-self.benchmark_weight_series).T.dot(self.estimated_alpha_cov).dot(w_tilde-self.benchmark_weight_series))
        # print('adjust_term', adjust_term)
        # print(self.estimated_alpha_cov)
        adjust_alpha = self.alpha_series - adjust_term * self.estimated_alpha_cov.dot(w_tilde-self.benchmark_weight_series)
        # print(adjust_alpha)

        self.param['alpha_series'] = adjust_alpha

        solve_result = MinRiskSolver(self.param).solve_without_round()
        return solve_result


# class RobustMinRiskSolver(MinRiskSolver):
#     def __init__(self, param):
#         super(RobustMinRiskSolver, self).__init__(param)
#         self.param = param
#         self.estimated_alpha_cov = param['estimated_alpha_cov']
#         self.kappa = param['kappa']
#
#     def solve_without_round(self):
#         w_c = cvx.Variable(self.alpha_series.size, nonneg=True)
#         # t = cvx.Variable(1, nonneg=True)
#         # 目标函数
#         ret = np.array(self.alpha_series.values)
#         bm = np.array(self.benchmark_weight_series.values)
#         objective = cvx.Minimize(cvx.quad_form((w_c - bm), self.asset_ret_cov))
#         # 股票权重 + 现金权重 = 1
#         sum_cons = cvx.sum(w_c) == 1
#         # alpha约束，带estimated error
#         # estimated_error_cons = cvx.SOC((ret.T * w_c - self.target_return), self.kappa * np.nan_to_num(np.sqrt(self.estimated_alpha_cov), 0.) @ w_c)
#         estimated_error_cons = ret.T * w_c - self.kappa * cvx.norm(cvx.matmul(np.linalg.cholesky(self.estimated_alpha_cov), w_c), 2) >= self.target_return
#         # ret_cons = ret.T * w_c >= self.target_return
#
#         asset_lower_bounds_cons = w_c >= self.asset_lower_boundary_series.values
#         asset_upper_bounds_cons = w_c <= self.asset_upper_boundary_series.values
#         constraints = [sum_cons, estimated_error_cons]
#         if asset_lower_bounds_cons is not None:
#             constraints += [asset_lower_bounds_cons]
#         if asset_upper_bounds_cons is not None:
#             constraints += [asset_upper_bounds_cons]
#
#         for el in constraints:
#             assert (el.is_dcp())
#
#         prob = cvx.Problem(objective, constraints)
#         prob.solve(solver=self.solver, verbose=False)
#
#         # 返回结果
#         after_trade_value = self.market_value * pd.Series(index=self.alpha_series.index, data=w_c.value)
#         after_trade_volume = np.round(after_trade_value / self.current_price)
#         after_trade_volume.fillna(0, inplace=True)
#         after_trade_ratio = after_trade_value / after_trade_value.sum()
#         exante_risk = np.sqrt(after_trade_ratio.T.dot(self.asset_ret_cov).dot(after_trade_ratio))
#         exante_return = after_trade_ratio.T.dot(ret)
#         exante_sr = exante_return / exante_risk
#         solve_result = {
#             "expected_weight": after_trade_ratio,
#             "expected_holding": after_trade_value,
#             "expected_volume": after_trade_volume,
#             "exante_return": exante_return,
#             "exante_risk": exante_risk,
#             "exante_sr": exante_sr,
#             "opt_status": prob.status
#         }
#         return solve_result

def simple_example():
    equity_list = ['A', 'B']
    alpha_true = pd.Series(index=equity_list, data=[2.48, 2.42])
    alpha_true['cash'] = 0.
    alpha_2 = pd.Series(index=equity_list, data=[2.5, 2.4])
    alpha_2['cash'] = 0.
    alpha_1 = pd.Series(index=equity_list, data=[2.4, 2.5])
    alpha_1['cash'] = 0.
    sigma_A, sigma_B = 0.42, 0.33
    rho_AB = 0.7
    cov_mat = [
        [sigma_A ** 2, rho_AB * sigma_A * sigma_B],
        [rho_AB * sigma_A * sigma_B, sigma_B ** 2]
    ]
    cov_df = pd.DataFrame(index=equity_list, columns=equity_list, data=cov_mat)

    cov_df = cov_df.reindex(equity_list + ['cash']).fillna(0.)
    cov_df['cash'] = 0.
    market_value = 1.
    current_price = pd.Series(index=equity_list + ['cash'], data=1.)

    target_risk = 0.1
    target_return = 0.2
    benchmark_weight = pd.Series(index=equity_list, data=[0.5, 0.5])
    benchmark_weight['cash'] = 0.
    asset_lower_boundary_series = pd.Series(index=equity_list, data=0.)
    asset_lower_boundary_series['cash'] = 0.
    asset_upper_boundary_series = pd.Series(index=equity_list, data=1.)
    asset_upper_boundary_series['cash'] = 0.
    param = {
        'asset_list': equity_list,
        'alpha_series': alpha_true,  # estimated alpha
        'asset_ret_cov': cov_df,
        'market_value': market_value,
        'current_price': current_price,
        'benchmark_weight_series': benchmark_weight,
        'asset_lower_boundary_series': asset_lower_boundary_series,
        'asset_upper_boundary_series': asset_upper_boundary_series,
        'target_risk': target_risk,
        'target_return': target_return,
        'solver': cvx.MOSEK,
        'kappa': 5,
        'estimated_alpha_cov': np.identity(cov_df.shape[0])
    }
    # print(MinRiskSolver(param).solve_without_round())
    print(RobustMaxReturnSolver(param).solve_without_round()['expected_weight'])
    # print(MaxReturnSolver(param).solve_without_round()['expected_weight'])

def real_example():
    cov_df = pd.read_csv('../cov.csv', index_col=0)
    alpha_df = pd.read_csv('../alpha.csv', header=None, names=['sec', 'alpha'])
    alpha_true = alpha_df.set_index('sec')['alpha']
    equity_list = alpha_true.index.tolist()
    equity_list = sorted(equity_list[:10])
    alpha_true = alpha_true.reindex(equity_list)
    cov_df = cov_df.loc[equity_list, equity_list]

    # equity_list.remove('cash')


    market_value = 1.
    current_price = pd.Series(index=equity_list, data=1.)

    benchmark_weight_series = pd.Series(index=equity_list, data=1./len(equity_list))
    # benchmark_weight_series['cash'] = 0.
    asset_lower_boundary_series = pd.Series(index=equity_list, data=0.)
    # asset_lower_boundary_series['cash'] = 0.
    asset_upper_boundary_series = pd.Series(index=equity_list, data=1.)
    # asset_upper_boundary_series['cash'] = 0.

    target_risk = 0.20
    target_return = 0.1
    kappa_list = [0, 0.5, 0.8, 1, 1.3, 1.8]
    lamb = 0.025

    # expected_weight_sum = None
    exante_return_sum = None
    experiment_time = 1
    period = 100

    param = {
        'asset_list': equity_list,
        'alpha_series': alpha_true,  # estimated alpha
        'asset_ret_cov': cov_df,
        'market_value': market_value,
        'current_price': current_price,
        'solver': cvx.MOSEK,
        'benchmark_weight_series': benchmark_weight_series,
        'asset_lower_boundary_series': asset_lower_boundary_series,
        'asset_upper_boundary_series': asset_upper_boundary_series,
        'target_risk': target_risk,

    }
    true_result = MaxReturnSolver(param).solve_without_round()
    true_expected_weight = true_result['expected_weight']
    true_expected_weight.name = 'true'

    # np.random.seed(1)
    for i in range(experiment_time):
        print('experiment no. %s' % i)
        exante_return_df = pd.DataFrame(index=kappa_list, columns=['alpha1', 'alpha2'])
        weight_corr_df = pd.DataFrame(index=kappa_list, columns=['alpha1', 'alpha2'])
        expected_weight_df = pd.DataFrame()

        data1 = np.random.multivariate_normal(alpha_true, cov_df, size=period)
        error1 = data1.T - np.expand_dims(alpha_true, axis=1)
        # sample_cov1 = np.cov(data1.T)
        error_cov1 = np.cov(error1)
        alpha_1 = pd.Series(index=equity_list, data=data1.mean(axis=0))  # estimate alpha

        data2 = np.random.multivariate_normal(alpha_true, cov_df, size=period)
        error2 = data2.T - np.expand_dims(alpha_true, axis=1)
        # sample_cov2 = np.cov(data2.T)
        error_cov2 = np.cov(error2)
        alpha_2 = pd.Series(index=equity_list, data=data2.mean(axis=0))  # estimate alpha
        for kappa in kappa_list:
            param = {
                'asset_list': equity_list,
                'alpha_series': alpha_1,  # estimated alpha
                'asset_ret_cov': cov_df,
                'market_value': market_value,
                'current_price': current_price,
                'solver': cvx.MOSEK,
                'benchmark_weight_series': benchmark_weight_series,
                'asset_lower_boundary_series': asset_lower_boundary_series,
                'asset_upper_boundary_series': asset_upper_boundary_series,
                'target_risk': target_risk,
                'target_return': target_return,
                'kappa': kappa,
                'estimated_alpha_cov': (1-lamb) * error_cov1 / period,
                'alpha_true': alpha_true
            }

            robust_result1 = RobustMaxReturnSolver(param).solve_without_round()

            param = {
                'asset_list': equity_list,
                'alpha_series': alpha_2,  # estimated alpha
                'asset_ret_cov': cov_df,
                'market_value': market_value,
                'current_price': current_price,
                'solver': cvx.MOSEK,
                'benchmark_weight_series': benchmark_weight_series,
                'asset_lower_boundary_series': asset_lower_boundary_series,
                'asset_upper_boundary_series': asset_upper_boundary_series,
                'target_risk': target_risk,
                'target_return': target_return,
                'kappa': kappa,
                'estimated_alpha_cov': (1-lamb) * error_cov2 / period,
                'alpha_true': alpha_true
            }

            robust_result2 = RobustMaxReturnSolver(param).solve_without_round()

            expected_weight1 = robust_result1['expected_weight']
            expected_weight2 = robust_result2['expected_weight']
            expected_weight1.name = '1.kappa_%s' % kappa
            expected_weight2.name = '2.kappa_%s' % kappa

            exante_return1 = expected_weight1.dot(alpha_true)
            exante_return2 = expected_weight2.dot(alpha_true)
            exante_risk1 = robust_result1['exante_risk']
            exante_risk2 = robust_result2['exante_risk']
            exante_return_df.loc[kappa, ['alpha1', 'alpha2']] = [exante_return1, exante_return2]
            expected_weight1_corr = np.corrcoef(expected_weight1, true_expected_weight)[0, 1]
            expected_weight2_corr = np.corrcoef(expected_weight2, true_expected_weight)[0, 1]
            weight_corr_df.loc[kappa, :] = [expected_weight1_corr, expected_weight2_corr]
            expected_weight_df = pd.concat([expected_weight_df, expected_weight1, expected_weight2], axis=1)

        if exante_return_sum is None:
            exante_return_sum = exante_return_df
        else:
            exante_return_sum += exante_return_df
    exante_return_sum /= experiment_time

    expected_weight_df = pd.concat([expected_weight_df, true_expected_weight], axis=1)

    pd.set_option('display.max_rows', expected_weight_df.shape[0])
    pd.set_option('display.max_columns', expected_weight_df.shape[1])
    print(expected_weight_df)
    print(true_expected_weight.sort_index())

    print(exante_return_sum)
    print(true_result['exante_return'])

    print(weight_corr_df)

    # print('robust_result1', robust_result1['expected_weight'])
    # print('robust_result2', robust_result2['expected_weight'])

if __name__ == '__main__':
    real_example()