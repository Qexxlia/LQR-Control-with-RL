import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO
from stable_baselines3.common.monitor import Monitor

from drone_env import DroneEnv as de

np.set_printoptions(precision=5, linewidth=10000)


def map_action(x):
    q_off_diagonal_mask = np.ones((12, 12), dtype=bool)
    q_off_diagonal_mask[np.triu_indices(12)] = False
    q_diagonal_mask = np.eye(12, dtype=bool)

    r_off_diagonal_mask = np.ones((4, 4), dtype=bool)
    r_off_diagonal_mask[np.triu_indices(4)] = False
    r_diagonal_mask = np.eye(4, dtype=bool)
    q_off_diagonal_weights = x[0:66]

    q_diagonal_weights = x[66:78]

    r_off_diagonal_weights = x[78:84]

    r_diagonal_weights = x[84:88]

    q = np.zeros((12, 12))
    q[q_off_diagonal_mask] = q_off_diagonal_weights / 1000
    q[q_diagonal_mask] = q_diagonal_weights

    r = np.zeros((4, 4))
    r[r_off_diagonal_mask] = r_off_diagonal_weights / 1000
    r[r_diagonal_mask] = r_diagonal_weights

    Q = q @ q.T
    R = r @ r.T
    min_svd = np.linalg.svd(R, compute_uv=False)[-1]
    if min_svd == 0 or min_svd < np.spacing(1.0) * np.linalg.norm(R, 1):
        return None, None

    return Q, R


env_params = {
    "max_duration": 150,
    "map_limits": np.array([-500, 500], dtype=np.float64),
    "t_step": 1,
    "seed": 0,
}

env = de(verbose=False, args=env_params)
env = Monitor(env)
obs = env.reset()
print(obs)

model = PPO.load(
    "./models/drone/checks/PPO_20240819-1719_1000/final_model/final_model.zip",
    env=env,
)

print("Simulating...")

# Initial Action
[action, _state] = model.predict(obs, deterministic=True)

# Record all actions taken
action_record = action

# Simulate
done = False
while not done:
    [obs, reward, done, info] = env.step(action)
    [action, _state] = model.predict(obs, deterministic=True)

    action_record = np.vstack((action_record, action))

    if info[-1].get("t", [0])[-1] > env_params.get("max_duration"):
        done = True

# actions = np.zeros((np.shape(action_record)[0]))
# for i in range(0, np.shape(action_record)[0]):
#     actions.append(vec_env.env_method("map_action", action))

# Extract data
position = info[-1].get("position", np.zeros(1))
attitude = info[-1].get("attitude", np.zeros(1))
velocity = info[-1].get("velocity", np.zeros(1))
spin = info[-1].get("spin", np.zeros(1))
t = info[-1].get("t", np.zeros(1))
u = info[-1].get("u", np.zeros(1))
reward = reward[-1]

print("Done")
print("Converged in: ", t[-1], "s")

print("Plotting...")

# Font sizes
tsL = 12
tsS = 7


def create_plot(fig_num, title, xlabel, ylabel, x_data, y_data, labels):
    fig = plt.figure(fig_num, figsize=(10, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_data, y_data, label=labels)
    ax.set_title(title, fontsize=tsL)
    ax.set_xlabel(xlabel, fontsize=tsS)
    ax.set_ylabel(ylabel, fontsize=tsS)
    if labels is not None:
        fig.legend()
    return fig


t3 = np.vstack((t, t, t)).transpose()

# Voltage
fig0 = create_plot(
    0, "Control", "Time (s)", "Control", t, u.transpose(), ["U1", "U2", "U3", "U4"]
)

# Position
fig1 = create_plot(
    1, "Position", "Time (s)", "Distance (m)", t3, position.transpose(), ["x", "y", "z"]
)

# Attitude
fig2 = create_plot(
    2,
    "Attitude",
    "Time (s)",
    "Angle (rad)",
    t3,
    attitude.transpose(),
    ["roll", "pitch", "yaw"],
)

# Velocity
fig3 = create_plot(
    3,
    "Velocity",
    "Time (s)",
    "Velocity (m/s)",
    t3,
    velocity.transpose(),
    ["x", "y", "z"],
)

# Spin
fig4 = create_plot(
    4,
    "Spin",
    "Time (s)",
    "Spin (rad/s)",
    t3,
    spin.transpose(),
    ["roll", "pitch", "yaw"],
)

plt.show()
print("Done")
