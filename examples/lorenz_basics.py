from systems.lorenz import Lorenz
from integration.explicit import rk4
import matplotlib.pyplot as plt

import sys

sys = Lorenz()
t, x, _ = rk4(sys, x0=[1, 1, 1], t_span=(0, 50), dt=0.001)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=0.5)
plt.show()

sys.exit()