import cvxpy as cvx
import numpy as np
import pandas as pd


class MaxReturnSolver:
    def __init__(self, param):
        self.asset_list = param['asset_list']
        self.alpha_series = param['alpha_series']
        self.asset_ret_cov = param['asset_ret_cov']
        self.market_value = param['market_value']
        self.current_price = param['current_price']
        self.benchmark_weight_series = param['benchmark_weight_series']
        self.asset_lower_boundary_series = param['asset_lower_boundary_series']
        self.asset_upper_boundary_series = param['asset_upper_boundary_series']
        self.target_risk = param['target_risk']
        self.solver = param['solver']

    def solve_without_round(self):
        w_c = cvx.Variable(self.alpha_series.size, nonneg=True)
        # 目标函数
        ret = np.array(self.alpha_series.values)
        objective = cvx.Maximize(ret.T * w_c)
        # 股票权重 + 现金权重 = 1
        sum_cons = cvx.sum(w_c) == 1
        # 风险约束
        risk_cons = cvx.quad_form((w_c - self.benchmark_weight_series), self.asset_ret_cov) <= self.target_risk ** 2
        asset_lower_bounds_cons = w_c >= self.asset_lower_boundary_series.values
        asset_upper_bounds_cons = w_c <= self.asset_upper_boundary_series.values

        constraints = [sum_cons, risk_cons]
        if asset_lower_bounds_cons is not None:
            constraints += [asset_lower_bounds_cons]
        if asset_upper_bounds_cons is not None:
            constraints += [asset_upper_bounds_cons]

        for el in constraints:
            assert (el.is_dcp())

        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=True)

        # 返回结果
        after_trade_value = self.market_value * pd.Series(index=self.alpha_series.index, data=w_c.value)
        after_trade_volume = np.round(after_trade_value / self.current_price)
        after_trade_volume.fillna(0, inplace=True)
        after_trade_ratio = after_trade_value / after_trade_value.sum()
        exante_risk = np.sqrt(after_trade_ratio.T.dot(self.asset_ret_cov).dot(after_trade_ratio))
        exante_return = after_trade_ratio.T.dot(ret)
        exante_sr = exante_return / exante_risk
        solve_result = {
            "expected_weight": after_trade_ratio,
            "expected_holding": after_trade_value,
            "expected_volume": after_trade_volume,
            "exante_return": exante_return,
            "exante_risk": exante_risk,
            "exante_sr": exante_sr,
            "opt_status": prob.status
        }
        return solve_result


class MinRiskSolver:
    def __init__(self, param):
        self.asset_list = param['asset_list']
        self.alpha_series = param['alpha_series']
        self.asset_ret_cov = param['asset_ret_cov']
        self.market_value = param['market_value']
        self.current_price = param['current_price']
        self.benchmark_weight_series = param['benchmark_weight_series']
        self.asset_lower_boundary_series = param['asset_lower_boundary_series']
        self.asset_upper_boundary_series = param['asset_upper_boundary_series']
        self.target_return = param['target_return']
        self.solver = param['solver']

    def solve_without_round(self):
        print(self.asset_upper_boundary_series)
        print(self.asset_lower_boundary_series)
        w_c = cvx.Variable(self.alpha_series.size, nonneg=True)
        # 目标函数
        bm = np.array(self.benchmark_weight_series.values)
        objective = cvx.Minimize(cvx.quad_form((w_c - bm), self.asset_ret_cov))
        # 股票权重 + 现金权重 = 1
        sum_cons = cvx.sum(w_c) == 1
        # 预期收益率约束
        ret = np.array(self.alpha_series.values)
        ret_cons = ret.T * w_c >= self.target_return

        constraints = [sum_cons, ret_cons]

        asset_lower_bounds_cons = w_c >= self.asset_lower_boundary_series.values
        asset_upper_bounds_cons = w_c <= self.asset_upper_boundary_series.values

        if asset_lower_bounds_cons is not None:
            constraints += [asset_lower_bounds_cons]
        if asset_upper_bounds_cons is not None:
            constraints += [asset_upper_bounds_cons]

        for el in constraints:
            assert (el.is_dcp())

        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.MOSEK, verbose=True)
        # 返回结果
        after_trade_value = self.market_value * pd.Series(index=self.alpha_series.index, data=w_c.value)
        after_trade_volume = np.round(after_trade_value / self.current_price)
        after_trade_volume.fillna(0, inplace=True)
        after_trade_ratio = after_trade_value / after_trade_value.sum()
        exante_risk = np.sqrt(after_trade_ratio.T.dot(self.asset_ret_cov).dot(after_trade_ratio))
        exante_return = after_trade_ratio.T.dot(ret)
        exante_sr = exante_return / exante_risk
        solve_result = {
            "expected_weight": after_trade_ratio,
            "expected_holding": after_trade_value,
            "expected_volume": after_trade_volume,
            "exante_return": exante_return,
            "exante_risk": exante_risk,
            "exante_sr": exante_sr,
            "opt_status": prob.status
        }
        return solve_result


class MaxSharpeRatioSolver:
    def __init__(self, param):
        self.alpha_series = param['alpha_series']
        self.market_value = param['market_value']
        self.asset_ret_cov = param['asset_ret_cov']
        self.current_price = param['current_price']
        self.benchmark_weight_series = param['benchmark_weight_series']
        self.asset_lower_boundary_series = param['asset_lower_boundary_series']
        self.asset_upper_boundary_series = param['asset_upper_boundary_series']
        self.cash_position = param['cash_position']

    def solve_without_round(self):
        w_c = cvx.Variable(self.alpha_series.size, nonneg=True)
        eta = cvx.Variable(1, nonneg=True)
        # 目标函数
        bm = np.array(self.benchmark_weight_series.values)
        objective = cvx.Minimize(cvx.quad_form((w_c - bm), self.asset_ret_cov))
        # 股票权重 + 现金权重 = 1
        sum_cons = cvx.sum(w_c) == 1.
        # unit alpha约束
        ret = np.array(self.alpha_series.values)
        unit_alpha_cons = ret.T * w_c == eta
        constraints = [sum_cons, unit_alpha_cons]

        asset_lower_bounds_cons = w_c >= self.asset_lower_boundary_series.values
        asset_upper_bounds_cons = w_c <= self.asset_upper_boundary_series.values
        if asset_lower_bounds_cons is not None:
            constraints += [asset_lower_bounds_cons]
        if asset_upper_bounds_cons is not None:
            constraints += [asset_upper_bounds_cons]

        for el in constraints:
            assert (el.is_dcp())

        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.MOSEK, verbose=True)

        # 返回结果
        after_trade_value = self.market_value * pd.Series(index=self.alpha_series.index, data=w_c.value)
        print('w_c', w_c.value)
        after_trade_volume = np.round(after_trade_value / self.current_price)
        after_trade_volume.fillna(0, inplace=True)
        after_trade_ratio = after_trade_value / after_trade_value.sum()
        exante_risk = np.sqrt(after_trade_ratio.T.dot(self.asset_ret_cov).dot(after_trade_ratio))
        exante_return = after_trade_ratio.T.dot(ret)
        exante_sr = exante_return / exante_risk
        solve_result = {
            "expected_weight": after_trade_ratio,
            "expected_holding": after_trade_value,
            "expected_volume": after_trade_volume,
            "exante_return": exante_return,
            "exante_risk": exante_risk,
            "exante_sr": exante_sr,
            "opt_status": prob.status
        }
        return solve_result
