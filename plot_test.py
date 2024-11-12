import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cylinder parameters
radius = 1
height = 5
resolution = 50  # resolution of the mesh

# Generate data for the cylinder
z = np.linspace(0, height, resolution)
theta = np.linspace(0, 2 * np.pi, resolution)
theta, z = np.meshgrid(theta, z)
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot for the cylinder
ax.plot_surface(x, y, z, color='lightblue', edgecolor='k', alpha=0.7)

# Add labels
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("3D Plot of a Solid Cylinder")

# Display the plot
plt.show()

