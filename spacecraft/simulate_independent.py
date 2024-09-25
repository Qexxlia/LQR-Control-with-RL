import matplotlib.pyplot as plt
import numpy as np

import spacecraft_dynamics as scd

plt.rcParams["text.usetex"] = True

time_range = (0, 1000)
t0 = time_range[0]

A, B = scd.precalcMatrices(6371 + 500, 3.986e5)

state = np.array(
    [
        0.5,  # x
        -0.5,  # y
        0,  # z
        1e-3,  # x_dot
        -1e-3,  # y_dot
        0,  # z_dot
        30,  # mass
    ],
    dtype=np.float64,
)

fig, ax = plt.subplots(1, 1, figsize=(9, 3))
plt.subplots_adjust(bottom=0.15)

satellite_mass = 15

q_weights = np.array([1, 1, 1, 1, 1, 1])
r_weights = np.ones(3)

sol1 = scd.simulate(state, time_range, q_weights, r_weights, A, B, 1e-3, satellite_mass)
ax.plot(sol1.t, sol1.y[0, :], label=r"$q_0=1$")
print(sol1.t[-1])

q_weights = np.array([100, 1, 1, 1, 1, 1])
r_weights = np.ones(3)

sol2 = scd.simulate(state, time_range, q_weights, r_weights, A, B, 1e-3, satellite_mass)
ax.plot(sol2.t, sol2.y[0, :], label=r"$q_0=100$")
print(sol2.t[-1])

ax.set_title(r"Impact of $q_0$ on LQR control")
ax.set_ylabel("$x$ distance (km)")
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")

plt.show()
