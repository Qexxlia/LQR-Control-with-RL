import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.linalg import solve_continuous_are

# Lqr controller for the drone dynamics
# Taken from Simulation of the Quadcopter Dynamics with LQR based control, Faraz Ahmad et al, 2018
#
# Yaw is phi
# Pitch is theta
# Roll is psi
#
# STATE VECTOR of form
#
# [
#     x,
#     y,
#     z,
#     psi,
#     theta,
#     phi,
#     x_dot,
#     y_dot,
#     z_dot,
#     psi_dot,
#     theta_dot,
#     phi_dot
# ]


## CALCULATIONS
def precalcMatrices(m, Ix, Iy, Iz):
    g = 9.81

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -g, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / m, 0, 0, 0],
            [0, 1 / Ix, 0, 0],
            [0, 0, 1 / Iy, 0],
            [0, 0, 0, 1 / Iz],
        ],
        dtype=np.float32,
    )

    return A, B


def matrices(qWeights, rWeights):
    Q = np.diag(qWeights)
    R = np.diag(rWeights)

    return Q, R


def calculateControl(state, u_max, K):
    # Calculate control
    u = -K @ state

    # Cap control
    u_magnitude = np.linalg.norm(u)
    if u_magnitude > u_max:
        u = (u / u_magnitude) * u_max

    return u


def nextState(t, state, q_weights, r_weights, A, B, K, u_max):
    # Get control
    u = calculateControl(state, u_max, K)

    # Calculate change
    delta_state = (A @ state) + (B @ u)
    return delta_state


## SIMULATION INTEGRATION
def simulate(state, time_range, q_weights, r_weights, A, B, u_max):

    input()
    [Q, R] = matrices(q_weights, r_weights)
    input()
    # [K, S, E] = ctrl.lqr(A, B, Q, R)
    [K, S, E] = lqr(A, B, Q, R)
    input()

    sol = integrate.solve_ivp(
        nextState,
        time_range,
        state,
        # max_step = 0.25,
        args=(q_weights, r_weights, A, B, K, u_max),
        events=(convergeEvent),
        atol=1e-6,
        rtol=1e-3,
    )

    return sol


def lqr(A, B, Q, R):
    X = solve_continuous_are(A, B, Q, R)
    K = scipy.linalg.inv(R) @ B.T @ X


## EVENTS
def convergeEvent(t, state, q_weights, r_weights, A, B, K, u_max):
    position_tol = 1e-3
    attitude_tol = 1e-3
    velocity_tol = 1e-6
    spin_tol = 1e-6

    tol = np.array(
        [
            position_tol,
            position_tol,
            position_tol,
            attitude_tol,
            attitude_tol,
            attitude_tol,
            velocity_tol,
            velocity_tol,
            velocity_tol,
            spin_tol,
            spin_tol,
            spin_tol,
        ]
    )

    exit = 0
    for i in range(12):
        if abs(state[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1


convergeEvent.terminal = True
convergeEvent.direction = 0


def flipEvent(t, state, q_weights, r_weights, A, B, K, u_max):
    for i in range(3, 6):
        if abs(state[i]) > np.pi / 8:
            return 0
    return 1


flipEvent.terminal = True
flipEvent.direction = 0
