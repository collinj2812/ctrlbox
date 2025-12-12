import numpy as np
from systems.base import ODESystem

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class InvertedPendulum(ODESystem):
    def default_params(self):
        return {'M': 1.0, 'm': 0.1, 'l': 0.5, 'g': 9.81, 'mu_c': 0.01, 'mu_p': 0.04}

    def _get_n_states(self):
        return 4

    def _get_n_inputs(self):
        return 1

    def rhs(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        M, m, l, g, mu_c, mu_p = self.params['M'], self.params['m'], self.params['l'], self.params['g'], self.params['mu_c'], self.params['mu_p']

        return np.array([
            x[1],
            (u[0] + m * l * x[3] ** 2 * np.sin(x[2]) - mu_c * np.sign(x[1])) / (M + m),
            x[3],
            (g * np.sin(x[2] - np.cos(x[2]) * (u[0] + m * l * x[3] ** 2 * np.sin(x[2])) / (M + m)) - mu_p * x[3]) / (l * (4/3 - m * np.cos(x[2]) ** 2 / (M + m)))
        ])

    def jacobian(self, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        pass

# animation
def animate_pendulum(t, x, l, u=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5)

    cart_width, cart_height = 0.3, 0.15
    cart = patches.Rectangle((0, 0), cart_width, cart_height, fc='steelblue')
    ax.add_patch(cart)

    rod, = ax.plot([], [], 'k', lw=2)
    bob, = ax.plot([], [], 'ko', markersize=15)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # arrow for input force
    u_scale = 0.01  # scale factor: arrow length per unit force
    arrow = ax.arrow(0, 0, 0.1, 0, head_width=0.05, head_length=0.02, fc='red', ec='red')

    def update(frame):
        cart_x = x[frame, 0]
        theta = x[frame, 2]

        cart.set_xy((cart_x - cart_width / 2, -cart_height / 2))

        bob_x = cart_x + l * np.sin(theta)
        bob_y = l * np.cos(theta)

        rod.set_data([cart_x, bob_x], [0, bob_y])
        bob.set_data([bob_x], [bob_y])
        time_text.set_text(f't = {t[frame]:.2f} s')

        # update arrow
        nonlocal arrow
        arrow.remove()
        if u is not None and frame < len(u):
            u_val = u[frame, 0] if u.ndim > 1 else u[frame]
            arrow_len = u_val * u_scale
            arrow = ax.arrow(cart_x, 0, arrow_len, 0,
                             head_width=0.05, head_length=0.02,
                             fc='red', ec='red')
        else:
            # invisible arrow
            arrow = ax.arrow(0, 0, 0, 0, head_width=0, head_length=0)

        return cart, rod, bob, time_text, arrow

    dt = t[1] - t[0]
    ani = FuncAnimation(fig, update, frames=len(t) - 1, interval=1000, blit=True)

    if save_path:
        ani.save(save_path, fps=1 / dt, progress_callback=lambda i, n: print(f'{i}/{n}', end='\r'))

    return ani