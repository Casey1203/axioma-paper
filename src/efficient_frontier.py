from .solver import *
from .rbo import *



class EfficientFrontier:
    def __init__(self, param):
        # self.target_risk = np.inf
        # self.target_ret = -np.inf
        self.asset_list = param['asset_list']
        self.benchmark_weight_series = param['benchmark_weight_series']
        self.alpha_series = param['alpha_series']
        self.asset_ret_cov = param['asset_ret_cov']
        self.market_value = param['market_value']
        self.current_price = param['current_price']
        self.solver = param['solver']
        self.target_risk = param['target_risk']
        self.target_return = param['target_return']
        self.asset_lower_boundary_series = param['asset_lower_boundary_series']
        self.asset_upper_boundary_series = param['asset_upper_boundary_series']

        # asset_lower_boundary_series = pd.Series(index=self.asset_list, data=-np.inf)
        # asset_lower_boundary_series['cash'] = 0.
        # asset_upper_boundary_series = pd.Series(index=self.asset_list, data=np.inf)
        # asset_upper_boundary_series['cash'] = 0.
        self.param = {
            'asset_list': self.asset_list,
            'alpha_series': self.alpha_series,
            'asset_ret_cov': self.asset_ret_cov,
            'market_value': self.market_value,
            'current_price': self.current_price,
            'benchmark_weight_series': self.benchmark_weight_series,
            'asset_lower_boundary_series': self.asset_lower_boundary_series,
            'asset_upper_boundary_series': self.asset_upper_boundary_series,
            'target_risk': self.target_risk,
            'target_return': self.target_return,
            'solver': cvx.MOSEK,
            # 'cash_position': 0.
        }

        self.min_return = None
        self.max_return = None

    def set_alpha_series(self, alpha_series):
        self.param['alpha_series'] = alpha_series

    def initialize(self):
        min_risk_result = MinRiskSolver(self.param).solve_without_round() # min risk
        max_return_result = MaxReturnSolver(self.param).solve_without_round() # max return
        # msr_result = self._initialize_sr(spo_input)

        # min
        if 'optimal' not in min_risk_result['opt_status']:
            min_risk_result['expected_weight'] = pd.Series(index=self.asset_list , data=0.)
            min_ret = self.param['alpha_series'].mean()
        else:
            min_ret = min_risk_result['expected_weight'].dot(self.param['alpha_series'])
            min_risk = min_risk_result['exante_risk']

        # max
        if 'optimal' not in max_return_result['opt_status']:
            max_return_result['expected_weight'] = pd.Series(index=self.asset_list , data=0.)
            max_ret = self.param['alpha_series'].max()
        else:
            max_ret = max_return_result['expected_weight'].dot(self.param['alpha_series'])
            max_risk = max_return_result['exante_risk']
        alpha_series_min = self.param['alpha_series'].min()
        alpha_series_max = self.param['alpha_series'].max()
        # if min_ret > alpha_series_min:
        #     min_ret = alpha_series_min
        # if max_ret <= alpha_series_max:
        #     max_ret = alpha_series_max
        self.min_return = min_ret
        self.max_return = max_ret
        self.min_risk = min_risk
        self.max_risk = max_risk

    # def _initialize_sr(self, spo_input):
    #     # max return
    #     msr_trade_result = MaxSharpeRatioSolver(spo_input).solve_without_round()
    #     return msr_trade_result

    def get_efficient_frontier(self, resolution=20, min_return=None, max_return=None):
        if min_return is not None:
            self.min_return = min_return
        if max_return is not None:
            self.max_return = max_return

        if self.min_return is None or self.max_return is None:
            msg = 'please run initialize first'
            raise Exception(msg)
        opt_result_list = []
        # delta = ((self.max_return - self.min_return) / resolution) - 1e-8
        delta = ((self.max_risk - self.min_risk) / resolution) - 1e-8
        # return_list = [(self.min_return + delta * i) for i in range(0, int(resolution)+1)]
        risk_list = [(self.min_risk + delta * i) for i in range(0, int(resolution) + 1)]
        param = self.param.copy()

        # for annual_ret_thresh in return_list:
        #     # print(annual_ret_thresh)
        #     param['target_return'] = annual_ret_thresh
        #     min_risk_dict = MinRiskSolver(param).solve_without_round()
        #
        #     if min_risk_dict['opt_status'] == 'optimal':  # 只保存有解的持仓
        #         opt_result_list.append(min_risk_dict)
        #     else:
        #         msg = \
        #             "compute efficient frontier warning: no solution under condition target_ret = %s" % (
        #                 annual_ret_thresh
        #             )
        #         print(msg)
        for annual_risk_thresh in risk_list:
            # print(annual_ret_thresh)
            param['target_risk'] = annual_risk_thresh
            max_return_dict = MaxReturnSolver(param).solve_without_round()

            if max_return_dict['opt_status'] == 'optimal':  # 只保存有解的持仓
                opt_result_list.append(max_return_dict)
            else:
                msg = \
                    "compute efficient frontier warning: no solution under condition target_risk = %s" % (
                        annual_risk_thresh
                    )
                print(msg)
        opt_result_list = sorted(opt_result_list, key=lambda x: x['exante_return'])
        return opt_result_list

class RobustEfficientFrontier:
    def __init__(self, param):
        # self.target_risk = np.inf
        # self.target_return = -np.inf
        self.asset_list = param['asset_list']
        # self.benchmark_weight_series = pd.Series(index=self.asset_list, data=0.)
        self.benchmark_weight_series = param['benchmark_weight_series']
        self.alpha_series = param['alpha_series']
        self.asset_ret_cov = param['asset_ret_cov']
        self.market_value = param['market_value']
        self.current_price = param['current_price']
        self.solver = param['solver']
        self.kappa = param['kappa']
        self.sqrt_Q = param['sqrt_Q']
        self.asset_lower_boundary_series = param['asset_lower_boundary_series']
        self.asset_upper_boundary_series = param['asset_upper_boundary_series']
        self.target_risk = param['target_risk']
        self.target_return = param['target_return']

        # asset_lower_boundary_series = pd.Series(index=self.asset_list, data=-np.inf)
        # asset_lower_boundary_series['cash'] = 0.
        # asset_upper_boundary_series = pd.Series(index=self.asset_list, data=np.inf)
        # asset_upper_boundary_series['cash'] = 0.
        self.param = {
            'asset_list': self.asset_list,
            'alpha_series': self.alpha_series,
            'asset_ret_cov': self.asset_ret_cov,
            'market_value': self.market_value,
            'current_price': self.current_price,
            'benchmark_weight_series': self.benchmark_weight_series,
            'asset_lower_boundary_series': self.asset_lower_boundary_series,
            'asset_upper_boundary_series': self.asset_upper_boundary_series,
            'target_risk': self.target_risk,
            'target_return': self.target_return,
            'solver': cvx.MOSEK,
            'kappa': self.kappa,
            'sqrt_Q': self.sqrt_Q
        }

        self.min_return = None
        self.max_return = None

    def set_alpha_series(self, alpha_series):
        self.param['alpha_series'] = alpha_series

    def initialize(self):
        min_risk_result = RobustMinRiskSolver(self.param).solve_without_round() # min risk
        max_return_result = RobustMaxReturnSolver(self.param).solve_without_round() # max return
        # msr_result = self._initialize_sr(spo_input)
        # min
        if 'optimal' not in min_risk_result['opt_status']:
            min_risk_result['expected_weight'] = pd.Series(index=self.asset_list, data=0.)
            min_ret = self.param['alpha_series'].mean()
        else:
            norm = np.linalg.norm(self.sqrt_Q.dot(min_risk_result['expected_weight']), 2)
            # print(norm)
            min_ret = min_risk_result['expected_weight'].dot(self.param['alpha_series']) - self.kappa * norm
            min_risk = min_risk_result['exante_risk']

        # max
        if 'optimal' not in max_return_result['opt_status']:
            max_return_result['expected_weight'] = pd.Series(index=self.asset_list, data=0.)
            max_ret = self.param['alpha_series'].max()
        else:
            norm = np.linalg.norm(self.sqrt_Q.dot(max_return_result['expected_weight']), 2)
            # print(norm)
            max_ret = max_return_result['expected_weight'].dot(self.param['alpha_series']) - self.kappa * norm
            max_risk = max_return_result['exante_risk']
        alpha_series_min = self.param['alpha_series'].min()
        alpha_series_max = self.param['alpha_series'].max()
        # if min_ret > alpha_series_min:
        #     min_ret = alpha_series_min
        # if max_ret <= alpha_series_max:
        #     max_ret = alpha_series_max
        self.min_return = min_ret
        self.max_return = max_ret
        self.min_risk = min_risk
        self.max_risk = max_risk

    # def _initialize_sr(self, spo_input):
    #     # max return
    #     msr_trade_result = MaxSharpeRatioSolver(spo_input).solve_without_round()
    #     return msr_trade_result

    def get_efficient_frontier(self, resolution=20, min_return=None, max_return=None):
        if min_return is not None:
            self.min_return = min_return
        if max_return is not None:
            self.max_return = max_return

        if self.min_return is None or self.max_return is None:
            msg = 'please run initialize first'
            raise Exception(msg)
        opt_result_list = []
        # delta = ((self.max_return - self.min_return) / resolution) - 1e-8
        delta = ((self.max_risk - self.min_risk) / resolution) - 1e-8
        # return_list = [(self.min_return + delta * i) for i in range(0, int(resolution)+1)]
        risk_list = [(self.min_risk + delta * i) for i in range(0, int(resolution) + 1)]
        param = self.param.copy()
        # for annual_ret_thresh in return_list:
        #     # print(annual_ret_thresh)
        #     param['target_return'] = annual_ret_thresh
        #     min_risk_dict = RobustMinRiskSolver(param).solve_without_round()
        #
        #     if min_risk_dict['opt_status'] == 'optimal':  # 只保存有解的持仓
        #         opt_result_list.append(min_risk_dict)
        #     else:
        #         msg = \
        #             "compute efficient frontier warning: no solution under condition target_ret = %s" % (
        #                 annual_ret_thresh
        #             )
        #         print(msg)

        for annual_risk_thresh in risk_list:
            # print(annual_ret_thresh)
            param['target_risk'] = annual_risk_thresh
            max_return_dict = RobustMaxReturnSolver(param).solve_without_round()

            if max_return_dict['opt_status'] == 'optimal':  # 只保存有解的持仓
                opt_result_list.append(max_return_dict)
            else:
                msg = \
                    "compute efficient frontier warning: no solution under condition target_risk = %s" % (
                        annual_risk_thresh
                    )
                print(msg)
        opt_result_list = sorted(opt_result_list, key=lambda x: x['exante_return'])
        return opt_result_list


class ShowEfficientFrontier:
    def __init__(self, portfolio_list, alpha_series, asset_ret_cov):
        self.portfolio_list = portfolio_list
        self.alpha_series = alpha_series
        self.asset_ret_cov = asset_ret_cov

    def show_frontier(self):
        result_list = []
        for portfolio in self.portfolio_list:
            expected_return = portfolio.dot(self.alpha_series)
            expected_risk = np.sqrt(portfolio.T.dot(self.asset_ret_cov).dot(portfolio))
            result_list.append({'exante_return': expected_return, 'exante_risk': expected_risk})
        return result_list
