import numpy as np

from systems.base import ODESystem

class Lorenz(ODESystem):
    def default_params(self):
        return {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}

    def _get_n_states(self):
        return 3

    def _get_n_inputs(self):
        return 0

    def rhs(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        s, r, b = self.params['sigma'], self.params['rho'], self.params['beta']

        return np.array([
            s * (x[1] - x[0]),
            x[0] * (r - x[2]) - x[1],
            x[0] * x[1] - b * x[2]
        ])

    def jacobian(self, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        s, r, b = self.params['sigma'], self.params['rho'], self.params['beta']

        return np.array([
            [-s, s, 0],
            [r - x[2], -1, -x[0]],
            [x[1], x[0], -b]
        ])
