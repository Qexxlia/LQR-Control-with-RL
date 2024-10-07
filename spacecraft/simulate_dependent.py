import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO
plt.rcParams["text.usetex"] = True
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

def map_range(val, in_min, in_max, out_min, out_max):
        return out_min * (out_max / out_min) ** (
            (val - in_min) / (in_max - in_min)
        )  # LOG

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

weights = np.zeros(action_record.shape)
t_steps = np.zeros(action_record.shape[0] + 1)
i = 0

for action in action_record:
	t_steps[i] = t_steps[i-1]+10
	weights[i] = map_range(action, -1, 1, 1e0, 1e6)
	i += 1
t_steps[i] = t_steps[i-1]+10
	
q_weights = weights[:, 0:6]
r_weights = weights[:, 6:9]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

ax1.plot(t, position[0, :], label=r"$x$")
ax1.plot(t, position[1, :], label=r"$y$")
ax1.plot(t, position[2, :], label=r"$z$")

ax2.plot(t, velocity[0, :], label=r"$x$")
ax2.plot(t, velocity[1, :], label=r"$y$")
ax2.plot(t, velocity[2, :], label=r"$z$")

fig1.suptitle(r"Spacecraft Rendezvous State-Dependent Control Performance")
for i in range(0, math.ceil(t[-1]), 20):
    ax1.axvspan(i, i + 10, color="lightgrey", alpha=0.5)
ax1.set_title(r"Position")
ax1.set_ylabel(r"Distance (km)")
ax1.set_xlabel(r"Time (s)")
ax1.legend(loc="upper right")

for i in range(0, math.ceil(t[-1]), 20):
    ax2.axvspan(i, i + 10, color="lightgrey", alpha=0.5)
ax2.set_title(r"Velocity")
ax2.set_ylabel(r"Velocity (km/s)")
ax2.set_xlabel(r"Time (s)")

fig2, ax3 = plt.subplots(1, 1, figsize=(9, 3))
ax3.plot(t, u.T[0, :], label=r"$u_x$")
ax3.plot(t, u.T[1, :], label=r"$u_y$")
ax3.plot(t, u.T[2, :], label=r"$u_z$")


fig2.suptitle(r"Spacecraft Rendezvous State-Dependent Control Efforts")
for i in range(0, math.ceil(t[-1]), 20):
    ax3.axvspan(i, i + 10, color="lightgrey", alpha=0.5)
ax3.set_ylabel(r"Control (km/s$^2$)")
ax3.set_xlabel(r"Time (s)")
ax3.legend(loc="lower right")

fig3, (ax4,ax5) = plt.subplots(1, 2, figsize=(9, 3))
ax4.stairs(q_weights[:, 0], t_steps, label=r"$q_0$", baseline=None)
ax4.stairs(q_weights[:, 1], t_steps, label=r"$q_1$", baseline=None)
ax4.stairs(q_weights[:, 2], t_steps, label=r"$q_2$", baseline=None)
ax4.stairs(q_weights[:, 3], t_steps, label=r"$q_3$", baseline=None)
ax4.stairs(q_weights[:, 4], t_steps, label=r"$q_4$", baseline=None)
ax4.stairs(q_weights[:, 5], t_steps, label=r"$q_5$", baseline=None)
ax5.stairs(r_weights[:, 0], t_steps, label=r"$r_0$", baseline=None)
ax5.stairs(r_weights[:, 1], t_steps, label=r"$r_1$", baseline=None)
ax5.stairs(r_weights[:, 2], t_steps, label=r"$r_2$", baseline=None)

fig3.suptitle(r"Spacecraft Rendezvous State-Dependent Weights")
ax4.set_ylabel(r"Weight")
ax4.set_yscale('log')
ax4.set_xlabel(r"Time (s)")

# Shrink current axis's height by 10% on the bottom
box = ax4.get_position()
ax4.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=6)

ax5.set_ylabel(r"Weight")
ax5.set_yscale('log')
ax5.set_xlabel(r"Time (s)")
# Shrink current axis's height by 10% on the bottom
box = ax5.get_position()
ax5.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=6)



fig1.tight_layout()
fig2.tight_layout()
# fig3.tight_layout()
fig1.savefig(
    "CW_state_dependent_state.pdf",
    format="pdf",
)
fig2.savefig(
    "CW_state_dependent_control.pdf",
    format="pdf",
)
fig3.savefig(
    "CW_state_dependent_weights.pdf",
    format="pdf",
)
plt.show()
