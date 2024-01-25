# SymMePlot
This package plots objects from the `sympy.physics.mechanics` module in `matplotlib`.

## Usage
Most of your programs are expected to follow this structure:
1. Creation of the system in sympy using the objects from `sympy.physics.mechanics`.
2. Create a figure with a 3D axes with `matplotlib`.
3. Initiate a `Scene` with the inertial frame and absolute origin.
4. Add your frames, vectors and points to the plotter instance.
5. Lambdify and evaluate the system.
6. Plot the system.

Below is a basic example of how this looks in practise:
```python
from sympy.physics.mechanics import Point, ReferenceFrame, dynamicsymbols
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt

# Create the system in sympy
N = ReferenceFrame('N')
A = ReferenceFrame('A')
q = dynamicsymbols('q')
A.orient_axis(N, N.z, q)
N0 = Point('N_0')
v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
A0 = N0.locatenew('A_0', v)
# Create the instance of the scene specifying the inertial frame and origin
scene = Scene3D(N, N0, scale=0.5)
# Add the objects to the system
scene.add_vector(v)
scene.add_frame(A, A0, ls='--')
scene.add_point(A0, color='g')
# Evaluate the system.
scene.lambdify_system(q)
scene.evaluate_system(0.5)
# Plot the system
scene.plot()
plt.show()

# You can also animate this system.
import numpy as np
from matplotlib.animation import FuncAnimation

# Setup the system for faster evaluation
scene.lambdify_system((q,))

def update(qi):
    scene.evaluate_system(qi)
    scene.update()
    return scene.artists


ani = FuncAnimation(fig, update, frames=np.linspace(0.5, 0.5 + 2 * np.pi, 100),
                    blit=True)
ani.save('animation.gif', fps=100)
```
