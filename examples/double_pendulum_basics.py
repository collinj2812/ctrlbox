from systems.double_inverted_pendulum import  DoubleInvertedPendulum, animate_double_pendulum
from integration.explicit import rk4
import matplotlib.pyplot as plt

import numpy as np
import sys

system = DoubleInvertedPendulum()

t_span = (0, 5)
dt = 0.02
n_steps = int((t_span[1] - t_span[0]) / dt) + 1
u = np.zeros((n_steps, 1))
x0=[-0, 0.9, 0, 0, 0, -30]

t, x, _ = rk4(system, x0=x0, u=u, t_span=t_span, dt=dt)

ani = animate_double_pendulum(t, x, system.default_params()['l1'], system.default_params()['l2'], 'double_pendulum.gif')

sys.exit()