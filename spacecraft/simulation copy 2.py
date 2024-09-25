import os

import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO

import spacecraft_dynamics as scd
from spacecraft_env import SpacecraftEnv as de

# Environment Parameters
env_params = {
    "variance_type": "none",
    "variance": 0,
    "max_duration": 40000,
    "map_limits": np.array(
        [
            [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
            [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        ],
        dtype=np.float64,
    ),
    "t_step_limits": np.array([10, 10], dtype=np.float64),
    "u_max": 1e-5,
    "simulation_type": "qrt",
    "t_step": 0,
    "seed": 0,
    "absolute_norm": True,
}

env = de(verbose=False, args=env_params)

model = PPO.load(
    os.getcwd()
    + "/models/spacecraft/integrate/PPO_20240716-0026/final_model/final_model.zip",
    env=env,
)

env.set_episode_options({"deterministic": 1, "verbose": 0})
[obs, _nfo] = env.reset()

[action, _state] = model.predict(obs, deterministic=True)
action_record = action

done = False
while not done:
    [obs, reward, done, done2, info] = env.step(action)
    [action, state_dep] = model.predict(obs, deterministic=True)
    action_record = np.vstack((action_record, action))

# Extract data
position = info.get("pos")
velocity = info.get("vel")
t = info.get("t")
mass = info.get("mass")

# PLOT DATA

# Font sizes
tsL = 12
tsS = 10

time_range = (0, 40000)
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
    dtype=np.float32,
)

satellite_mass = 15

q_weights = np.array(
    [
        2.92293472e01,
        2.04151647e01,
        4.63774004e01,
        1.61606450e05,
        5.69216350e05,
        7.17820101e05,
    ]
)

r_weights = [9.80987258e05, 3.85048729e05, 4.06807113e05]

sol_independent = scd.simulate(
    state, time_range, q_weights, r_weights, A, B, 1e-5, satellite_mass
)

fig, ax = plt.subplots(2, 1)

ax[0].set_title("Spacecraft Controller Performance\nu_max = 10mm/s^2\n\nX Position")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("x (km)")
# ax[0].set_xlim(0, 1500)
ax[0].plot(sol_independent.t, sol_independent.y[0, :])
ax[0].plot(t, position[0])

ax[1].set_title("Y Position")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("y (km)")
# ax[1].set_xlim(0, 1500)
ax[1].plot(sol_independent.t, sol_independent.y[1, :])
ax[1].plot(t, position[1])

plt.tight_layout()
fig.legend(["PSO", "RL"])

print(sol_independent.t[-1])
print(t[-1])

print(sol_independent.y[-1, -1] - state[-1])
print(mass - state[-1])

plt.show()
fig.savefig("plot.pdf")
