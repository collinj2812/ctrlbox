from systems.inverted_pendulum import InvertedPendulum, animate_pendulum
from integration.explicit import rk4
from control.lqr import LQRController
import matplotlib.pyplot as plt

import numpy as np
import sys

system = InvertedPendulum()

# calculate jacobian around x0 and u0
system.setup_jacobian_num(x0=np.array([1, 0, 2 * np.pi, 0]), u0=np.array([0]))

# first set angle to 2 pi
angle_set = 2 * np.pi
controller = LQRController(A=system.jac_x, B=system.jac_u, Q=np.diag([1e-7,1e-15,1e-4,1e-18]), R=np.array([[1e-5]]), N=np.array([1,1,1e-5,1]) * 0, setpoint=np.array([1, 0, 2 * np.pi, 0]))

print(f'Eigenvalues of A_BK: {np.linalg.eigvals(system.jac_x-system.jac_u@(- controller.R_inv @ (controller.B.T @ controller.P + controller.N.T)))}')

t_span = (0, 30)
dt = 0.001
n_steps = int((t_span[1] - t_span[0]) / dt) + 1
t = np.linspace(*t_span, n_steps + 1)
x_0 = [1, 0, 2 * np.pi, 0]

Q = np.diag([0, 0, 0, 1]) * 1e-3

x = np.zeros((n_steps + 1, system._get_n_states()))
u = np.zeros((n_steps, system._get_n_inputs()))
x[0] = x_0
for step_i in range(n_steps):
    u[step_i] = controller(x[step_i], dt)

    x[step_i + 1], _ = rk4(system, x=x[step_i], u=u[step_i], dt=dt)
    x[step_i + 1] += np.random.multivariate_normal(np.zeros(system._get_n_states()), Q )

    # disturbance
    # if step_i == int(n_steps / 6):
    #     x[step_i + 1, 1] += 0.4

max_fps = 10
skip = max(1, int(1 / (dt * max_fps)))
ani = animate_pendulum(t[::skip], x[::skip], system.default_params()['l'], u[::skip], './animations/pendulum_lqr.gif')

sys.exit()