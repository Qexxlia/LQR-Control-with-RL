import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO
plt.rcParams["text.usetex"] = True
plt.rcParams.update({'font.size': 22})
plt.rcParams['lines.linewidth'] = 3.0

from spacecraft_env_verbose import SpacecraftEnv as de
import spacecraft_dynamics_verbose as scd

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


print("dependent:")
print(t[-1])
mass = info.get("mass")

delta_v = 316 * 9.81 * math.log(30/mass)
print("DV")
print(delta_v)


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
        30],
    dtype=np.float64,
)

satellite_mass = 15

q_weights1 = np.array([8.65372e+01, 1.66806e+01, 5.07821e+05, 6.30032e+05, 4.30389e+05, 1.30445e+03])
r_weights1 = np.array([6.79712e+05, 2.76023e+05, 5.52777e+05])

sol, u = scd.simulate(state, time_range, q_weights1, r_weights1, A, B, 1e-5, satellite_mass)

mass = sol.y[6,-1]

print("independent:")
print(sol.t[-1])
delta_v = 3300 * 9.81 * math.log(30/mass)
print("DV")
print(delta_v)

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.plot(sol.t, sol.y[0,:], label = r"State-independent")
ax2.plot(sol.t, sol.y[1,:], label = r"State-independent")

ax1.plot(t, position[0, :], label=r"State-dependent", linestyle="dashed")
ax2.plot(t, position[1, :], label=r"State-dependent", linestyle="dashed")


fig1.suptitle(r"Spacecraft Rendezvous Controller Performance")
ax1.set_title(r"$x$ Position")
# ax1.set_ylabel(r"Distance (km)")
# ax1.set_xlabel(r"Time (s)")
ax1.legend(loc="best")

ax2.set_title(r"$y$ Position")
# ax2.set_ylabel(r"Distance (km)")
ax2.set_xlabel(r"Time (s)")
#ax2.legend(loc="upper right")

fig1.supylabel("Distance (km)")


fig2, (ax3,ax4) = plt.subplots(1, 2, figsize=(14, 5))
ax3.stairs(q_weights[:, 0], t_steps, label=r"$q_0$", baseline=None)
ax3.stairs(q_weights[:, 1], t_steps, label=r"$q_1$", baseline=None)
ax3.stairs(q_weights[:, 2], t_steps, label=r"$q_2$", baseline=None)
ax3.stairs(q_weights[:, 3], t_steps, label=r"$q_3$", baseline=None)
ax3.stairs(q_weights[:, 4], t_steps, label=r"$q_4$", baseline=None)
ax3.stairs(q_weights[:, 5], t_steps, label=r"$q_5$", baseline=None)
ax4.stairs(r_weights[:, 0], t_steps, label=r"$r_0$", baseline=None)
ax4.stairs(r_weights[:, 1], t_steps, label=r"$r_1$", baseline=None)
ax4.stairs(r_weights[:, 2], t_steps, label=r"$r_2$", baseline=None)

fig2.suptitle(r"Spacecraft Rendezvous State-Dependent Weights")
ax3.set_ylabel(r"Weight")
ax3.set_title(r"Q Weights")
ax3.set_yscale('log')
ax3.set_xlabel(r"Time (s)")

ax4.set_ylabel(r"Weight")
ax4.set_title(r"R Weights")
ax4.set_yscale('log')
ax4.set_xlabel(r"Time (s)")

fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(
    "poster-cw-performance.png",
    format="png",
)
fig2.savefig(
    "CW_state_dependent_weights.png",
    format="png",
)
plt.show()
