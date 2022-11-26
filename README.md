# SymMePlot
This package plots objects from the `sympy.physics.mechanics` module in `matplotlib`.

## Requirements
There is no requirements file yet, but this package currently relies on `sympy`, `numpy` and `matplotlib`. One should also note that I'm currently using `Python 3.10`.

## Usage
This can be done in the following steps:
1. Create your system in sympy using the objects from `sympy.physics.mechanics`.
2. Create a figure with a 3D axes with `matplotlib`.
3. Create an instance of `SymMePlotter` in which you define the inertial frame and absolute origin.
4. Add your frames, vectors and points the plotter instance.
5. Evaluate the system.
6. Plot the system.

Below is a basic example of how this looks in practise:
```python
from sympy.physics.mechanics import Point, ReferenceFrame, dynamicsymbols
from symmeplot import SymMePlotter
import matplotlib.pyplot as plt

# Create the system in sympy
N = ReferenceFrame('N')
A = ReferenceFrame('A')
q = dynamicsymbols('q')
A.orient_axis(N, N.z, q)
N0 = Point('N_0')
v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
A0 = N0.locatenew('A_0', v)
# Create the matplotlib 3d axes
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# Create the instance of the plotter specifying the inertial frame and origin
plotter = SymMePlotter(ax, N, N0, scale=0.5)
# Add the objects to the system
plotter.add_vector(v)
plotter.add_frame(A, A0, ls='--')
plotter.add_point(A0, color='g')
# Evaluate the system.
# This method is preferred if you like to evaluate the system once.
plotter.evalf(subs={q: 0.5})
# Plot the system
plotter.plot()
fig.show()

# You can also animate this system.
import numpy as np
from matplotlib.animation import FuncAnimation

# Setup the system for faster evaluation
plotter.lambdify_system((q,))

def update(qi):
    plotter.evaluate_system(qi)
    return plotter.update()


ani = FuncAnimation(fig, update, frames=np.linspace(0.5, 0.5 + 2 * np.pi, 100),
                    blit=True)
ani.save('animation.gif', fps=100)
```

