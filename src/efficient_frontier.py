from .solver import *

class EfficientFrontier:
    def __init__(self, param):
        cash_position = 0.
        self.benchmark_weight_series = pd.Series()
        self.target_risk = 99
        self.target_ret = -99
        self.asset_list = param['asset_list']
        self.alpha_series = param['alpha_series']
        self.asset_ret_cov = param['asset_ret_cov']
        self.market_value = param['market_value']
        self.current_price = param['current_price']
        self.benchmark_weight_series = param['benchmark_weight_series']
        self.solver = param['solver']

    def _initialize(self, spo_input):
        opt_result_list = []
        min_risk_result = self._initialize_min(spo_input)
        max_return_result = self._initialize_max(spo_input)
        msr_result = self._initialize_sr(spo_input)

        # min
        if 'optimal' not in min_risk_result['opt_status']:
            min_risk_result['expected_weight'] = pd.Series(index=self.asset_list + ['cash'], data=0.)
            min_ret = spo_input['alpha_series'].mean()
        else:
            min_ret = min_risk_result['expected_weight'].dot(spo_input['alpha_series'])
        opt_result_list.append(min_risk_result)

        # max
        if 'optimal' not in max_return_result['opt_status']:
            max_return_result['expected_weight'] = pd.Series(index=self.asset_list + ['cash'], data=0.)
            max_ret = spo_input['alpha_series'].max()
        else:
            max_ret = max_return_result['expected_weight'].dot(spo_input['alpha_series'])
        opt_result_list.append(max_return_result)

        if 'optimal' in msr_result['opt_status']:
            opt_result_list.append(msr_result)
        return opt_result_list, min_ret, max_ret

    def _initialize_min(self, spo_input):
        # min risk
        solve_result = MinRiskSolver(spo_input).solve_without_round()
        return solve_result

    def _initialize_max(self, spo_input):
        # max return
        solve_result = MaxReturnSolver(spo_input).solve_without_round()
        return solve_result

    def _initialize_sr(self, spo_input):
        # max return
        msr_trade_result = MaxSharpeRatioSolver(spo_input).solve_without_round()
        return msr_trade_result

    def get_efficient_frontier(self):
        asset_lower_boundary_series = pd.Series(index=self.asset_list, data=-np.inf)
        asset_lower_boundary_series['cash'] = 0.
        asset_upper_boundary_series = pd.Series(index=self.asset_list, data=np.inf)
        asset_upper_boundary_series['cash'] = 0.
        param = {
            'asset_list': self.asset_list,
            'alpha_series': self.alpha_series,
            'asset_ret_cov': self.asset_ret_cov,
            'market_value': self.market_value,
            'current_price': self.current_price,
            'benchmark_weight_series': self.benchmark_weight_series,
            'asset_lower_boundary_series': asset_lower_boundary_series,
            'asset_upper_boundary_series': asset_upper_boundary_series,
            'target_risk': 99,
            'target_return': -99,
            'solver': cvx.MOSEK,
            'cash_position': 0.
        }
        opt_result_list, min_return, max_return = self._initialize(param)
        delta = ((max_return - min_return) / 20.0) - 1e-8
        return_list = [(min_return + delta * i) for i in range(1, 20)]

        for annual_ret_thresh in return_list:
            param['target_return'] = annual_ret_thresh
            min_risk_dict = MinRiskSolver(param).solve_without_round()

            if min_risk_dict['opt_status']:  # 只保存有解的持仓
                opt_result_list.append(min_risk_dict)
            else:
                msg = \
                    "compute efficient frontier warning: no solution under condition target_ret = %s" % (
                        annual_ret_thresh
                    )
                print(msg)
        opt_result_list = sorted(opt_result_list, key=lambda x: x['exante_return'])
        return opt_result_list
