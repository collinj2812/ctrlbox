import numpy as np
import time

def rk4(sys, x0, u = None, t_span=(0, 1), dt = 0.1):
    # check inputs for consistency
    assert len(x0) == sys.n_states

    time_start = time.perf_counter()
    n_steps = int((t_span[1]-t_span[0]) / dt)
    t = np.linspace(*t_span, n_steps + 1)
    sol = np.zeros((n_steps + 1, sys.n_states))
    sol[0] = x0
    for step_i in range(1, n_steps + 1):
        t_i = step_i*dt + t_span[0]
        if u:
            k1 = sys.rhs(t_i, sol[step_i - 1], u[step_i - 1])
            k2 = sys.rhs(t_i + dt / 2, sol[step_i - 1] + dt * k1 / 2, u[step_i - 1])
            k3 = sys.rhs(t_i + dt / 2, sol[step_i - 1] + dt * k2 / 2, u[step_i - 1])
            k4 = sys.rhs(t_i + dt, sol[step_i - 1] + dt * k3, u[step_i - 1])
        else:
            k1 = sys.rhs(t_i, sol[step_i - 1])
            k2 = sys.rhs(t_i + dt / 2, sol[step_i - 1] + dt * k1 / 2)
            k3 = sys.rhs(t_i + dt / 2, sol[step_i - 1] + dt * k2 / 2)
            k4 = sys.rhs(t_i + dt, sol[step_i - 1] + dt * k3)

        sol[step_i] = sol[step_i - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    time_end = time.perf_counter()

    meta_data = {
        'comp_time': time_end - time_start,
        'n_steps': n_steps,
        't_span': t_span,
        'dt': dt
    }
    return t, sol, meta_data
