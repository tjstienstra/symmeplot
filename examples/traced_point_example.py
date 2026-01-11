"""Example demonstrating the traced point feature in symmeplot.

This example shows how to use the `add_point_trace` method to create
a chronophotography-like effect where a moving point leaves a trail
that fades over time.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from symmeplot.matplotlib import Scene3D

# Create symbolic variables
t = sm.symbols("t")

# Define reference frame and origin
N = me.ReferenceFrame("N")
O = me.Point("O")

# Create a point that moves in a 3D spiral
radius = 0.4
P = O.locatenew(
    "P",
    radius * sm.cos(t) * N.x + radius * sm.sin(t) * N.y + 0.1 * t * N.z
)

# Example 1: Basic traced point with alpha decay
print("Creating Example 1: Basic traced point with alpha decay")
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

scene1 = Scene3D(N, O, ax=ax1)
scene1.add_point_trace(
    P,
    name="Spiral Trace",
    frequency=1,  # Record every evaluation
    alpha_decay=lambda age: max(0.1, 1.0 - age / 30),  # Fade older points
    color="red",
    linewidths=3
)
scene1.add_frame(N, scale=0.2)

# Lambdify and evaluate
scene1.lambdify_system((t,))
for t_val in np.linspace(0, 4 * np.pi, 60):
    scene1.evaluate_system(t_val)
    scene1.update()

scene1.plot()
ax1.set_title("Traced Point with Alpha Decay")
plt.savefig("traced_point_example1.png", dpi=150, bbox_inches='tight')
print("Saved: traced_point_example1.png")

# Example 2: Comparison of different frequencies
print("\nCreating Example 2: Comparison of frequencies")
fig2, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

frequencies = [1, 3, 5]
colors = ['red', 'blue', 'green']

for ax, freq, color in zip(axes, frequencies, colors):
    scene = Scene3D(N, O, ax=ax)
    scene.add_point_trace(
        P,
        frequency=freq,
        alpha_decay=lambda age: max(0.2, 1.0 - age / 20),
        color=color,
        linewidths=2
    )
    scene.add_frame(N, scale=0.2)
    
    scene.lambdify_system((t,))
    for t_val in np.linspace(0, 4 * np.pi, 60):
        scene.evaluate_system(t_val)
        scene.update()
    
    scene.plot()
    ax.set_title(f"Frequency = {freq}")

plt.tight_layout()
plt.savefig("traced_point_example2.png", dpi=150, bbox_inches='tight')
print("Saved: traced_point_example2.png")

# Example 3: Animation
print("\nCreating Example 3: Animated traced point")
fig3 = plt.figure(figsize=(8, 8))
ax3 = fig3.add_subplot(111, projection='3d')

scene3 = Scene3D(N, O, ax=ax3)
scene3.add_point_trace(
    P,
    frequency=2,
    alpha_decay=lambda age: max(0.1, 1.0 - age / 25),
    color="purple",
    linewidths=3
)
scene3.add_frame(N, scale=0.2)
scene3.lambdify_system((t,))

# Create animation
ani = scene3.animate(
    lambda t_val: (t_val,),
    frames=np.linspace(0, 6 * np.pi, 120),
    interval=50
)

# Save animation
try:
    ani.save("traced_point_animation.gif", writer='pillow', fps=20)
    print("Saved: traced_point_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")

print("\nExample complete!")
print("\nUsage:")
print("  scene.add_point_trace(point, frequency=1, alpha_decay=None, color='blue', **kwargs)")
print("\nParameters:")
print("  - point: The Point or Vector to trace")
print("  - frequency: Record every Nth evaluation (default: 1)")
print("  - alpha_decay: Function(age) -> transparency (default: lambda _: 1.0)")
print("  - color: Color of the trace (default: 'blue')")
print("  - **kwargs: Additional arguments for LineCollection (e.g., linewidths)")
