import numpy as np
import scipy

class StateFeedbackController:
    def __init__(self, A, B, eigvals, setpoint):
        self.A = A
        self.B = B
        self.setpoint = setpoint
        self.K = scipy.signal.place_poles(self.A, self.B, eigvals).gain_matrix

    def __call__(self, x, dt) -> np.ndarray:
        u = self.K @ x
        return u
