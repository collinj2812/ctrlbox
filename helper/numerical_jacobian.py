import numpy as np

def numerical_jacobian(rhs, n_states, n_inputs, x0, u0, h=1e-6):
    # go through each output for each state and calculate difference of disturbed outputs
    jac_x = np.zeros((n_states, n_states))
    for out_i in range(n_states):
        upper = np.array([rhs(0, x0 + np.eye(n_states)[in_i] * h / 2, u0)[out_i] for in_i in
                          range(n_states)])
        lower = np.array([rhs(0, x0 - np.eye(n_states)[in_i] * h / 2, u0)[out_i] for in_i in
                          range(n_states)])
        jac_x[out_i] = (upper - lower) / h

    jac_u = np.zeros((n_states, n_inputs))
    for out_i in range(n_states):
        upper = np.array([rhs(0, x0, u0 + np.eye(n_inputs)[in_i] * h / 2)[out_i] for in_i in
                          range(n_inputs)])
        lower = np.array([rhs(0, x0, u0 - np.eye(n_inputs)[in_i] * h / 2)[out_i] for in_i in
                          range(n_inputs)])
        jac_u[out_i] = (upper - lower) / h

    return jac_x, jac_u