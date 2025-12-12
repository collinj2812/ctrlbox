import numpy as np


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = None

    def __call__(self, x, dt) -> np.ndarray:
        error = x - self.setpoint
        self.integral += error * dt

        if self.prev_error is None:
            derivative = 0
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return u

    def reset(self):
        self.integral = 0
        self.prev_error = None
