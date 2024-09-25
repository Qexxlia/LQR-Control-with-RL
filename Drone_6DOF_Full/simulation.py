import matplotlib.pyplot as plt
import numpy as np

import drone_dynamics as dd

np.set_printoptions(precision=5, linewidth=10000)

state = np.array(
    [
        10,
        7,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
)


A, B = dd.precalcMatrices()

time_range = (0, 150)


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


Q, R = map_action(
    np.array(
        [
            -460.51152,
            92.9935,
            59.93823,
            298.32354,
            -75.31625,
            -111.30947,
            17.79598,
            294.28849,
            477.18789,
            90.07883,
            188.47596,
            64.23828,
            73.17355,
            262.60728,
            68.16507,
            -496.69991,
            376.47317,
            59.67795,
            58.74123,
            479.69466,
            107.28669,
            -219.90335,
            -69.1725,
            98.32019,
            -14.31672,
            334.71391,
            117.87988,
            429.00644,
            25.78202,
            67.14969,
            -405.63305,
            -74.08007,
            -270.69486,
            -217.74847,
            126.98494,
            99.15227,
            186.51689,
            -115.23144,
            -107.45724,
            146.22356,
            -138.85334,
            -38.79658,
            78.62204,
            146.0864,
            123.40344,
            34.85803,
            14.48828,
            -130.4354,
            98.23999,
            187.05597,
            -177.62076,
            -486.07784,
            -263.39694,
            -278.67049,
            392.86344,
            10.37307,
            67.88431,
            290.3755,
            -50.38298,
            -102.24784,
            326.27324,
            -420.2245,
            347.72282,
            -270.23234,
            16.72728,
            158.3997,
            398.98522,
            -91.88751,
            435.28286,
            146.17682,
            402.45422,
            -475.9021,
            -179.9464,
            49.82745,
            -16.10471,
            -69.96286,
            -91.06682,
            -171.22569,
            146.59572,
            314.27315,
            77.484,
            -42.17137,
            -321.43276,
            404.07827,
            -38.0373,
            472.88911,
            -45.63711,
            122.94889,
        ]
    )
)

print(Q, R)

print("Simulating...")
sol, u = dd.simulate(state, time_range, Q, R, A, B)
print("Done")
print("Converged in: ", sol.t[-1], "s")

if len(sol.t_events[1]) != 0:
    print("Failed")

print("Plotting...")
position = sol.y[0:3]
attitude = sol.y[3:6]
velocity = sol.y[6:9]
spin = sol.y[9:12]

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


t3 = np.vstack((sol.t, sol.t, sol.t)).transpose()

# Voltage
fig0 = create_plot(
    0, "Control", "Time (s)", "Control", sol.t, u.transpose(), ["U1", "U2", "U3", "U4"]
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
