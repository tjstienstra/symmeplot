Welcome to symmeplot's documentation!
=====================================

SymMePlot is a visualization tool designed for mechanical systems created using the
mechanics module in SymPy_, :mod:`sympy.physics.mechanics`.

The :mod:`sympy.physics.mechanics` module allows users to define mechanical systems
symbolically to derive their analytic equations of motion. During this process, users
can construct various objects such as reference frames, points, bodies, and more.

SymMePlot enhances this process by providing a way to visualize these constructed
objects. It integrates with visualization backends like Matplotlib_, and creates visual
representations based on the parametrization of the symbols involved in the system.

To install :mod:`symmeplot` with the :mod:`matplotlib` visualization run: ::

   pip install symmeplot matplotlib

Most of your programs are expected to follow this structure:

1. Creation of the system in sympy using the objects from
   :mod:`sympy.physics.mechanics`.
2. Create a figure with a 3D axes with :mod:`matplotlib`.
3. Initiate a :class:`Scene<symmeplot.matplotlib.scene.Scene3D>` with the inertial frame
   and absolute origin.
4. Add your frames, vectors and points to the plotter instance.
5. Lambdify and evaluate the system.
6. Plot the system.

Here is a simple example of how it looks in practice: ::

   import numpy as np
   from symmeplot.matplotlib import Scene3D
   from sympy.physics.mechanics import Point, ReferenceFrame, dynamicsymbols

   # Create the system in sympy
   N = ReferenceFrame("N")
   A = ReferenceFrame("A")
   q = dynamicsymbols("q")
   A.orient_axis(N, N.z, q)
   N0 = Point("N_0")
   v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
   A0 = N0.locatenew("A_0", v)
   # Create the instance of the scene specifying the inertial frame and origin
   scene = Scene3D(N, N0, scale=0.5)
   # Add the objects to the system
   scene.add_vector(v)
   scene.add_frame(A, A0, ls="--")
   scene.add_point(A0, color="g")
   # Evaluate the system.
   scene.lambdify_system(q)
   scene.evaluate_system(0.5)
   # Plot the system
   scene.plot()

   # You can also animate this system.
   ani = scene.animate(lambda q: (q,), frames=np.linspace(0, 2 * np.pi, 60))
   ani.save("animation.gif", fps=30)

.. image:: animation.gif
   :alt: animation
   :align: center

Contents
========

.. toctree::
   :maxdepth: 2

   guides/installation
   api/index

.. _SymPy: https://www.sympy.org
.. _Matplotlib: https://matplotlib.org/
