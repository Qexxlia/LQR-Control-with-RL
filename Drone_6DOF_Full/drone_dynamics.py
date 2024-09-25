# Imports
# import control as ctrl

import math

import numpy as np
from mpmath import cos, sec, sin, tan
from scipy import integrate
from scipy.linalg import solve_continuous_are

"""
Lqr controller for the drone dynamics
Taken from Simulation of the Quadcopter Dynamics with LQR based control, Faraz Ahmad et al, 2018
"""

"""
Yaw is phi
Pitch is theta
Roll is psi

STATE VECTOR of form

[
    x,
    y,
    z,
    psi,
    theta,
    phi,
    x_dot,
    y_dot,
    z_dot,
    psi_dot,
    theta_dot,
    phi_dot
]
"""


# CALCULATIONS
def precalcMatrices():
    g = 9.81
    K_f = 0.1188
    K_t = 0.0036
    I_x = 0.0552
    I_y = 0.05525
    I_z = 0.1104
    L = 0.1969
    m = 0.547

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
        dtype=np.float64,
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
            [K_f / m, K_f / m, K_f / m, K_f / m],
            [0, 0, L * K_f / I_x, -L * K_f / I_x],
            [L * K_f / I_y, -L * K_f / I_y, 0, 0],
            [-K_t / I_z, -K_t / I_z, K_t / I_z, K_t / I_z],
        ],
        dtype=np.float64,
    )

    return A, B


def calculateControl(state, K):
    # Calculate control
    u = -K @ state
    u = np.clip(u, -24, 24)
    return u


def nextState(t, state, A, B, K):
    # tilt_max = math.radians(60)
    # if abs(state[3]) > tilt_max:
    #     state[3] = math.copysign(tilt_max, state[3])
    # if abs(state[4]) > tilt_max:
    #     state[4] = math.copysign(tilt_max, state[4])
    #     print(state)

    # state[3] = max(min(state[3], -tilt_max), tilt_max)
    # state[4] = max(min(state[4], -tilt_max), tilt_max)

    # Get control
    u = calculateControl(state, K)

    # Calculate change
    delta_state = (A @ state) + (B @ u)
    # delta_state = non_linearState(state, u)

    return delta_state


# SIMULATION INTEGRATION
def simulate(state, time_range, Q, R, A, B):

    try:
        K = lqr(A, B, Q, R)
    except Exception as e:
        # print("LQR Failed")
        # print("Q:", Q)
        # print("R:", R)
        # print("Det:", np.linalg.det(R))
        # print("Svd:", np.linalg.svd(R, compute_uv=False)[-1])
        # print("Spacing:", np.spacing(1.0) * np.linalg.norm(R, 1))
        print(e)
        return None, []

    sol = integrate.solve_ivp(
        nextState,
        time_range,
        state,
        # max_step = 0.25,
        args=(A, B, K),
        events=(convergeEvent, attitudeEvent),
        method="LSODA",
    )

    u = -K @ sol.y

    return sol, u


def lqr(A, B, Q, R):
    # Solve the continuous time algebraic Riccati equation
    X = solve_continuous_are(A, B, Q, R)

    # Compute the LQR gain
    K = np.linalg.inv(R) @ B.T @ X
    return K


# EVENTS
def convergeEvent(t, state, A, B, K):
    attitude_tol = 1e-3
    spin_tol = 1e-6
    position_tol = 1e-3
    velocity_tol = 1e-6

    position = state[0:3]
    attitude = state[3:6]
    velocity = state[6:9]
    spin = state[9:12]

    if (
        (abs(position) < position_tol).all()
        and (abs(attitude) < attitude_tol).all()
        and (abs(velocity) < velocity_tol).all()
        and (abs(spin) < spin_tol).all()
    ):
        return 0
    else:
        return 1


convergeEvent.terminal = True
convergeEvent.direction = 0


def attitudeEvent(t, state, A, B, K):
    tilt_max = math.radians(60)

    if (abs(state[3:4]) > tilt_max).any():
        return 0
    else:
        return 1


attitudeEvent.terminal = True
attitudeEvent.direction = 0
