from abc import ABC, abstractmethod
import numpy as np

class ODESystem(ABC):
    def __init__(self, params: dict = None):
        self.params = params or self.default_params()
        self.n_states = self._get_n_states()
        self.n_inputs = self._get_n_inputs()

    @abstractmethod
    def rhs(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        pass

    def output(self, x: np.ndarray) -> np.ndarray:
        return x

    def output_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.eye(self.n_states)