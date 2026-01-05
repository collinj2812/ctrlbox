from systems.inverted_pendulum import InvertedPendulum, animate_pendulum
from integration.explicit import rk4
from control.pid import PIDController
import matplotlib.pyplot as plt

import numpy as np
import sys
import time

system = InvertedPendulum()

# first set angle to 2 pi
angle_set = 2 * np.pi
controller = PIDController(10, 0, 1, angle_set)

t_span = (0, 30)
dt = 0.001
n_steps = int((t_span[1] - t_span[0]) / dt) + 1
t = np.linspace(*t_span, n_steps + 1)
x_0 = [1, 0, 2 * np.pi, 0]

Q = np.diag([0, 0, 0, 1]) * 1e-3

x = np.zeros((n_steps + 1, system._get_n_states()))
u = np.zeros((n_steps, system._get_n_inputs()))
x[0] = x_0
time_before = time.perf_counter()
for step_i in range(n_steps):
    u[step_i] = controller(x[step_i, 2], dt)

    x[step_i + 1], _ = rk4(system, x=x[step_i], u=u[step_i], dt=dt)
    x[step_i + 1] += np.random.multivariate_normal(np.zeros(system._get_n_states()), Q )

    # disturbance
    # if step_i == int(n_steps / 6):
    #     x[step_i + 1, 1] += 0.4
time_after = time.perf_counter()
print(f'Time for simulation:{time_after - time_before}')

max_fps = 10
skip = max(1, int(1 / (dt * max_fps)))
ani = animate_pendulum(t[::skip], x[::skip], system.default_params()['l'], u[::skip], './animations/pendulum_pid.gif')

sys.exit()


# x_cas = ca.SX.sym('x', system._get_n_states())
# u_cas = ca.SX.sym('u', system._get_n_inputs())
# rhs_cas = ca.SX(system.rhs(0, x_cas, u_cas))
# jac = ca.jacobian(rhs_cas,x_cas)
