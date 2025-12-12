from systems.inverted_pendulum import InvertedPendulum, animate_pendulum
from integration.explicit import rk4
from control.pid import PIDController
import matplotlib.pyplot as plt

import numpy as np
import sys

system = InvertedPendulum()
outer_controller = PIDController(-1e-6*0, -2e-6, -0.0004, 1)  # track position 0 and change angle for this

# first set angle to 2 pi
angle_set = 2 * np.pi
inner_controller = PIDController(10, 0, 1, angle_set)  # track angle from outer loop

t_span = (0, 30)
dt = 0.001
n_steps = int((t_span[1] - t_span[0]) / dt) + 1
t = np.linspace(*t_span, n_steps + 1)
x_0 = [1, 0, 3 * np.pi, 0]

Q = np.diag([0, 0, 0, 1e-4]) * 0
use_controller = False

x = np.zeros((n_steps + 1, system._get_n_states()))
u = np.zeros((n_steps, system._get_n_inputs()))
x[0] = x_0
for step_i in range(n_steps):

    if use_controller:
        # outer loop
        angle_set = angle_set + outer_controller(x[step_i, 0], dt)  # x[step_i, 0] is position
        inner_controller.setpoint = angle_set
        u[step_i] = inner_controller(x[step_i, 2], dt)  # x[step_i, 2] is angle
    else:
        # swing pendulum until it reaches upper equilibrium point
        swing_freq = 0.01
        if np.sin(step_i*swing_freq) > 0:
            u[step_i] = -1.0
        else:
            u[step_i] = 1.0

        # check if close to upright and turn on controller (check 2 pi and 4 pi)
        if np.fabs(x[step_i, 2] - 2 * np.pi) < 0.2:
            use_controller = True
            angle_set = 2 * np.pi
            inner_controller.setpoint = angle_set
        elif np.fabs(x[step_i, 2] - 4 * np.pi) < 0.2:
            use_controller = True
            angle_set = 4 * np.pi
            inner_controller.setpoint = angle_set


    x[step_i + 1], _ = rk4(system, x=x[step_i], u=u[step_i], dt=dt)
    x[step_i + 1] += np.random.multivariate_normal(np.zeros(system._get_n_states()), Q )

    # if step_i == int(n_steps / 6):
    #     x[step_i + 1, 1] += 0.4

max_fps = 10
skip = max(1, int(1 / (dt * max_fps)))
ani = animate_pendulum(t[::skip], x[::skip], system.default_params()['l'], u[::skip], 'pendulum.gif')

sys.exit()