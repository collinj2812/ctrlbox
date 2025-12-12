import numpy as np
import time

def rk4(sys, x, u = None, t=None, dt = 0.1):
    # check inputs for consistency
    assert len(x) == sys.n_states

    time_start = time.perf_counter()

    if t is None:
        t = 0

    if u is not None:
        k1 = sys.rhs(t, x, u)
        k2 = sys.rhs(t + dt / 2, x + dt * k1 / 2, u)
        k3 = sys.rhs(t + dt / 2, x + dt * k2 / 2, u)
        k4 = sys.rhs(t + dt, x + dt * k3, u)
    else:
        k1 = sys.rhs(t, x)
        k2 = sys.rhs(t + dt / 2, x + dt * k1 / 2)
        k3 = sys.rhs(t + dt / 2, x + dt * k2 / 2)
        k4 = sys.rhs(t + dt, x + dt * k3)

    sol = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    time_end = time.perf_counter()

    meta_data = {
        'comp_time': time_end - time_start,
        'dt': dt
    }
    return sol, meta_data
