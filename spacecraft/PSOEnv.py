import numpy as np
import pyswarms as ps

import spacecraft_dynamics as scd


def simulate(action):
    time_range = (0, 50000)

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
        dtype=np.float32,
    )

    satellite_mass = 15

    size = np.shape(action)[0]
    reward = np.zeros(size)

    # Simulate dynamics
    # print("\n")
    for i in range(0, size):

        q_weights = action[i, 0:6]
        r_weights = action[i, 6:9]

        sol = scd.simulate(
            state, time_range, q_weights, r_weights, A, B, 1e-5, satellite_mass
        )

        if sol is None:
            reward[i] = 1e9
            continue

        dVT = 0
        dVT -= sol.y[6, -1] - sol.y[6, 0]
        totalTime = sol.t[-1]

        reward[i] = totalTime**2 + dVT**2
    return reward


# Hyperparameters
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

# Bounds
bounds = (
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]),
)

# Optimizer
optimizer = ps.single.GlobalBestPSO(
    n_particles=1000, dimensions=9, options=options, bounds=bounds
)

cost, pos = optimizer.optimize(simulate, 100)
