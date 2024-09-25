import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO

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
    [action, _state] = model.predict(obs, deterministic=True)
    action_record = np.vstack((action_record, action))

# Extract data
position = info.get("pos")
velocity = info.get("vel")
t = info.get("t")

# PLOT DATA
t_e = np.vstack((t, t, t)).transpose()
t_e1 = np.vstack((t, t, t, t)).transpose()

# Font sizes
tsL = 12
tsS = 10

# Voltage
# fig3 = create_plot(3, 'Voltage', 'Time (s)', 'Voltage (V)', t_e1, u.transpose(), ['F','B','L','R'])

# Attitude
# fig4 = create_plot(5, 'Attitude', 'Time (s)', 'Angle (rad)', t_e, attitude.transpose(), ['roll','pitch', 'yaw'])

# Spin
# fig6 = create_plot(7, 'Spin', 'Time (s)', 'Spin (rad/s)', t_e, spin.transpose(), ['roll','pitch','yaw'])


fig, (ax1) = plt.subplots(1)
ax1.set_title(r"State-Dependent Spacecraft Position u_{max} = 10mm/s^2", fontsize=tsL)
line = ax1.plot(t_e, position.transpose(), label=["x", "y", "z"])
""" line[0].set_dashes([2,2,10,2])
line[0].set_dash_capstyle('round')
line[1].set_dashes([5,2,5,2])
line[1].set_dash_capstyle('round')
line[2].set_dashes([2,2,2,2])
line[2].set_dash_capstyle('round') """
for i in range(0, math.ceil(t[-1]), 20):
    ax1.axvspan(i, i + 10, color="lightgrey", alpha=0.5)
ax1.set_ylabel("Position (km)", fontsize=tsS)
ax1.set_xlabel("Time (s)", fontsize=tsS)

# line = ax2.plot(t_e, np.degrees(spin.transpose()), label=['_roll','_pitch', '_yaw'])
# line[0].set_dashes([2,2,10,2])
# line[0].set_dash_capstyle('round')
# line[1].set_dashes([5,2,5,2])
# line[1].set_dash_capstyle('round')
# line[2].set_dashes([2,2,2,2])
# line[2].set_dash_capstyle('round')
# ax2.set_ylabel('Spin (deg/s)', fontsize=tsS)
# ax2.axvspan(0, 0.1, color='lightgrey', alpha=0.5)
# ax2.axvspan(0.2, 0.3, color='lightgrey', alpha=0.5)
# ax2.axvspan(0.4, t[-1], color='lightgrey', alpha=0.5)
fig.legend(loc="outside right center")

plt.show()
