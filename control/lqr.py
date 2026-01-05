import numpy as np
import scipy

class LQRController:
    def __init__(self, A, B, Q, R, N, setpoint):
        self.P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)
        self.N = N
        self.setpoint = setpoint

    def __call__(self, x, dt) -> np.ndarray:
        u = - self.R_inv @ (self.B.T @ self.P + self.N.T) @ (x-self.setpoint)
        return u
