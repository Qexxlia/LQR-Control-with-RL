import numpy as np
import matplotlib.pyplot as plt
import spacecraft_dynamics as scd

state = np.array([
    0.5,    # x
    -0.5,    # y
    0,    # z
    1e-3,   # x_dot
    -1e-3,   # y_dot
    0,   # z_dot
    30,   # mass 
], dtype=np.float32)

q_weights = np.ones(6)
r_weights = np.ones(3)

[A,B] = scd.precalcMatrices(6371 + 500, 3.986e5)

u_max = 1e-2

sol = scd.simulate(state, (0, 1000), q_weights, r_weights, A, B, u_max, 15)

plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])
plt.show()

delta_v = sol.y[6, -1] - sol.y[6, 0]

print(sol.y[6])

time = sol.t[-1] - sol.t[0]

print("Delta V: " + str(delta_v))
print("Time: " + str(time))

input()