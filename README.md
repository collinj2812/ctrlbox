# ctrlbox

A personal laboratory for numerical methods, estimation, and control of nonlinear dynamical systems.

## Purpose

This repository serves as a living reference for implementing and understanding:

- **Numerical integration**: explicit and implicit schemes, adaptive methods
- **State estimation**: observers, Kalman filters (EKF, UKF), MHE
- **Control**: feedback linearization, LQR, MPC variants
- **System identification**: subspace methods, neural approaches, SINDy

All methods are tested on simple but richly nonlinear benchmark systems (Lorenz, Van der Pol, Rössler).

## Installation

```bash
git clone https://github.com/collinj2812/ctrlbox.git
cd ctrlbox
pip install -e .
```

## Quick Start

```python
from ctrlbox.systems import Lorenz
from ctrlbox.integrate import rk4
import matplotlib.pyplot as plt

sys = Lorenz()
t, x = rk4(sys, x0=[1, 1, 1], t_span=(0, 50), dt=0.01)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=0.5)
plt.show()
```

## Structure

```
ctrlbox/
├── systems.py      # ODE system definitions
├── integrate.py    # Numerical integrators
├── estimate.py     # State estimation methods
├── control.py      # Control algorithms
└── utils.py        # Plotting, metrics
```

## License

MIT
