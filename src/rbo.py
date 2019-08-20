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
        solve_result = super(RobustMaxReturnSolver).solve_without_round()
        w_tilde = solve_result['expected_weight']
        adjust_term = np.sqrt(self.kappa ** 2 / (w_tilde).T * self.estimated_alpha_cov * w_tilde)
        adjust_alpha = self.alpha_series - adjust_term * self.estimated_alpha_cov * w_tilde

        self.param['alpha_series'] = adjust_alpha

        super(RobustMaxReturnSolver).solve_without_round()

