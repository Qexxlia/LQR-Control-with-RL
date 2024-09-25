# Imports
# import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
    r, p, y, dr, dp, dy
]
"""


## CALCULATIONS
def precalcMatrices(Kt, Kf, L, Jr, Jp, Jy):
    g = 9.81

    A = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, L * Kf / Jr, -L * Kf / Jr],
            [L * Kf / Jp, -L * Kf / Jp, 0, 0],
            [-Kt / Jy, -Kt / Jy, Kt / Jy, Kt / Jy],
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
    # ex = False
    # for ctrl in u:
    #     if abs(ctrl) > u_max:
    #         ex = True
    #         break
    # if ex:
    #     u = u / max(abs(u)) * u_max
    np.clip(u, -u_max, u_max)

    return u


def nextState(t, state, q_weights, r_weights, A, B, K, u_max):
    # Get control
    u = calculateControl(state, u_max, K)

    # Calculate change
    delta_state = (A @ state) + (B @ u)
    return delta_state


## SIMULATION INTEGRATION
def simulate(state, time_range, q_weights, r_weights, A, B, u_max):

    [Q, R] = matrices(q_weights, r_weights)
    K = lqr(A, B, Q, R)

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
    return K


## EVENTS
def convergeEvent(t, state, q_weights, r_weights, A, B, K, u_max):
    attitude_tol = 0.0005
    spin_tol = 0.00005

    tol = np.array(
        [attitude_tol, attitude_tol, attitude_tol, spin_tol, spin_tol, spin_tol]
    )

    exit = 0
    for i in range(6):
        if abs(state[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1


convergeEvent.terminal = True
convergeEvent.direction = 0
