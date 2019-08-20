import cvxpy as cvx
import numpy as np
import pandas as pd
from .solver import *

class RobustMaxReturnSolver(MaxReturnSolver):
    def __init__(self, param):
        super(RobustMaxReturnSolver, self).__init__(param)
        self.param = param
        self.estimated_alpha_cov = param['estimated_alpha_cov']
        self.kappa = param['kappa']
        self.z = 0

    def solve_without_round(self):
        solve_result = MaxReturnSolver(self.param).solve_without_round()
        w_tilde = solve_result['expected_weight']
        print('w_tilde', w_tilde)
        adjust_term = np.sqrt(self.kappa ** 2 / (w_tilde).T.dot(self.estimated_alpha_cov).dot(w_tilde))
        # print('adjust_term', adjust_term)
        # print(self.estimated_alpha_cov)
        adjust_alpha = self.alpha_series - adjust_term * self.estimated_alpha_cov.dot(w_tilde)
        print(adjust_alpha)

        self.param['alpha_series'] = adjust_alpha

        solve_result = MaxReturnSolver(self.param).solve_without_round()
        return solve_result

