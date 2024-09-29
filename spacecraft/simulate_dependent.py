import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO

from spacecraft_env_verbose import SpacecraftEnv as de

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
    os.getcwd() + "/models/spacecraft/RESULTS/final/best_model/best_model.zip",
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
u = info.get("u")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

ax1.plot(t, position[0, :], label=r"$x$")
ax1.plot(t, position[1, :], label=r"$y$")
ax1.plot(t, position[2, :], label=r"$z$")

ax2.plot(t, velocity[0, :], label=r"$x$")
ax2.plot(t, velocity[1, :], label=r"$y$")
ax2.plot(t, velocity[2, :], label=r"$z$")

fig1.suptitle(r"Spacecraft Rendezvous State-Dependent Control Performance")
ax1.set_title(r"Position")
ax1.set_ylabel(r"Distance (km)")
ax1.set_xlabel(r"Time (s)")
ax1.legend(loc="upper right")
ax2.set_title(r"Velocity")
ax2.set_ylabel(r"Velocity (km/s)")
ax2.set_xlabel(r"Time (s)")

fig2, ax3 = plt.subplots(1, 1, figsize=(9, 3))
ax3.plot(t, u[0, :], label=r"$u_x$")
ax3.plot(t, u[1, :], label=r"$u_y$")
ax3.plot(t, u[2, :], label=r"$u_z$")

fig2.suptitle(r"Spacecraft Rendezvous State-Dependent Control Efforts")
ax3.set_ylabel(r"Control (km/s$^2$)")
ax3.set_xlabel(r"Time (s)")
ax3.legend(loc="lower right")

fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(
    "../../../Reports/Final Report/Figures/CW_state_dependent_state.pdf",
    format="pdf",
)
fig2.savefig(
    "../../../Reports/Final Report/Figures/CW_state_dependent_control.pdf",
    format="pdf",
)
plt.show()
