import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from spacecraft_env import SpacecraftEnv as se


def analysis(file, log):
    # Create the environment
    env_params = {
        "variance_type": "none",
        "variance": 0,
        "max_duration": 750,
        "map_limits": np.array(
            [
                [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
                [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3],
            ],
            dtype=np.float32,
        ),
        "t_step_limits": np.array([10, 10], dtype=np.float32),
        "u_max": 1e-2,
        "simulation_type": "qrt",
        "t_step": 0,
        "seed": 0,
        "absolute_norm": True,
    }
    env = se(verbose=False, args=env_params)

    # Load the model
    model = PPO.load(file, env=env)

    # Run the model
    env.set_episode_options({"deterministic": 1, "verbose": 0, "log": log})
    [obs, _info] = env.reset()
    [action, _state] = model.predict(obs, deterministic=True)
    [obs, reward, terminated, truncated, info] = env.step(action)

    while not terminated and not truncated:
        [action, _state] = model.predict(obs, deterministic=True)
        [obs, reward, terminated, truncated, info] = env.step(action)

    deltaV = env.deltaV
    time = env.current_time

    return deltaV, time


deltaV = [0, 0, 0, 0, 0, 0, 0]
time = [0, 0, 0, 0, 0, 0, 0]

deltaV[0], time[0] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2209/best_model/best_model",
    0,
)
deltaV[1], time[1] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2045/best_model/best_model",
    0,
)
deltaV[2], time[2] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2029/best_model/best_model",
    0,
)
deltaV[3], time[3] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2027_1/best_model/best_model",
    0,
)
deltaV[4], time[4] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2011/best_model/best_model",
    0,
)
deltaV[5], time[5] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2045_1/best_model/best_model",
    0,
)
deltaV[6], time[6] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240709-2208/best_model/best_model",
    0,
)


deltaV_L = [0, 0, 0, 0, 0]
time_L = [0, 0, 0, 0, 0]

deltaV_L[0], time_L[0] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240710-1138/best_model/best_model",
    1,
)
deltaV_L[1], time_L[1] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240710-1141/best_model/best_model",
    1,
)
deltaV_L[2], time_L[2] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240710-1141_1/best_model/best_model",
    1,
)
deltaV_L[3], time_L[3] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240710-1143/best_model/best_model",
    1,
)
deltaV_L[4], time_L[4] = analysis(
    os.getcwd()
    + "/models/spacecraft/data_collection/PPO_20240710-1142/best_model/best_model",
    1,
)

x = [1 / 1000, 1 / 100, 1 / 10, 1 / 1, 10 / 1, 100 / 1, 1000 / 1]
x_L = [1 / 100, 1 / 10, 1 / 1, 10 / 1, 100 / 1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, deltaV)
ax.plot(x_L, deltaV_L)
ax.set_xscale("log")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, time)
ax.plot(x_L, time_L)
ax.set_xscale("log")

plt.show()

