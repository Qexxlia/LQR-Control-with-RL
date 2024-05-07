import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the gravitational constant and the mass of the central body
G = 6.67430e-11  # m^3 kg^-1 s^-2
M = 1.989e30  # kg (mass of the sun)

# Define the initial position and velocity vectors
r = np.array([1.496e11, 0])  # m
v = np.array([0, 29.78e3])  # m/s

# Define the time step and the number of steps
dt = 60 * 60 * 24  # s
n_steps = 365

# Create a figure and an axes
fig, ax = plt.subplots()
dot, = ax.plot(r[0], r[1], 'ro')
line, = ax.plot([], [], 'b')

# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-2e11, 2e11), ylim=(-2e11, 2e11))
ax.grid()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global r, v

    # Calculate the acceleration
    a = -G * M * r / np.linalg.norm(r)**3

    # Update the velocity and position
    v += a * dt
    r += v * dt

    dot.set_data(r[0], r[1])
    xdata, ydata = line.get_data()
    line.set_data(np.append(xdata, r[0]), np.append(ydata, r[1]))

    return line,

ani = FuncAnimation(fig, update, frames=range(n_steps), init_func=init, blit=False)

plt.show()

ani.save('orbit.mp4', writer='ffmpeg')