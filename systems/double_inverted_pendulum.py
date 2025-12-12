import numpy as np
from systems.base import ODESystem

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class DoubleInvertedPendulum(ODESystem):
    def default_params(self):
        return {'M': 1.0, 'm1': 0.1, 'm2': 0.1, 'l1': 0.5, 'l2': 0.5, 'g': 9.81, 'mu_c': 0.01, 'mu_p': 0.04}

    def _get_n_states(self):
        return 6

    def _get_n_inputs(self):
        return 1

    def rhs(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        M, m1, m2, l1, l2, g, mu_c, mu_p = self.params['M'], self.params['m1'], self.params['m2'], self.params['l1'], self.params['l2'], self.params['g'], self.params['mu_c'], self.params['mu_p']
        pos, v, th1, om1, th2, om2 = x

        c1 = np.cos(th1)
        c2 = np.cos(th2)
        s1 = np.sin(th1)
        s2 = np.sin(th2)
        c12 = np.cos(th1 - th2)
        s12 = np.sin(th1 - th2)

        # mass matrix
        mass_matrix = np.array([
            [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
            [(m1 + m2) * l1 * c1, (m1 + m2) * l1 ** 2, m2 * l1 * l2 * c12],
            [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2 ** 2]
        ])

        # forces
        forces = np.array([
            (m1 + m2) * l1 * om1 ** 2 * s1 + m2 * l2 * om2 ** 2 * s2 + u[0] - mu_c * np.sign(v),
            -(m1 + m2) * g * s1 - m2 * l1 * l2 * om2 **2 * s12 - mu_p * om1,
            -m2 * g * s2 + m2 * l1 * l2 * om1 ** 2 * s12 - mu_p * om2
        ])

        accels = np.linalg.solve(mass_matrix, forces)

        return np.array([v, accels[0], om1, accels[1], om2, accels[2]])

    def jacobian(self, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        pass

# animation
def animate_double_pendulum(t, x, l1, l2, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5)

    cart_width, cart_height = 0.3, 0.15
    cart = patches.Rectangle((0, 0), cart_width, cart_height, fc='steelblue')
    ax.add_patch(cart)

    rod1, = ax.plot([], [], 'k', lw=2)
    rod2, = ax.plot([], [], 'k', lw=2)
    bob1, = ax.plot([], [], 'ko', markersize=12)
    bob2, = ax.plot([], [], 'ro', markersize=12)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        cart_x = x[frame, 0]
        th1 = x[frame, 2]
        th2 = x[frame, 4]  # assuming state order [pos, v, th1, om1, th2, om2]

        cart.set_xy((cart_x - cart_width / 2, -cart_height / 2))

        # first bob position
        bob1_x = cart_x + l1 * np.sin(th1)
        bob1_y = -l1 * np.cos(th1)

        # second bob position (relative to first bob)
        bob2_x = bob1_x + l2 * np.sin(th2)
        bob2_y = bob1_y - l2 * np.cos(th2)

        rod1.set_data([cart_x, bob1_x], [0, bob1_y])
        rod2.set_data([bob1_x, bob2_x], [bob1_y, bob2_y])
        bob1.set_data([bob1_x], [bob1_y])
        bob2.set_data([bob2_x], [bob2_y])
        time_text.set_text(f't = {t[frame]:.2f} s')

        return cart, rod1, rod2, bob1, bob2, time_text

    dt = t[1] - t[0]
    ani = FuncAnimation(fig, update, frames=len(t) - 1, interval=dt * 1000, blit=True)

    if save_path:
        ani.save(save_path, fps=30, progress_callback=lambda i, n: print(f'{i}/{n}', end='\r'))

    return ani