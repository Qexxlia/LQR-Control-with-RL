# Imports
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy import integrate

'''
State
X = [
    x,
    y,
    z,
    x_dot,
    y_dot,
    z_dot
]
'''

## CALCULATIONS
def precalcMatrices(a, mu):
    n = np.sqrt(mu/a**3)

    A = np.array(
        [
            [  0,   0,   0,   1,   0,   0],
            [  0,   0,   0,   0,   1,   0],
            [  0,   0,   0,   0,   0,   1],
            [3*n**2,   0,   0,   0, 2*n,   0],
            [  0,   0,   0, -2*n,   0,   0],
            [  0,   0, -n**2,   0,   0,   0]
        ], dtype=np.float64
    )

    B = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float64
    )
    return A, B

def matrices(qWeights, rWeights):
    Q = np.diag(qWeights)
    R = np.diag(rWeights)

    return Q, R

def calculateControl(state, u_max, A, B, K):
    # Calculate control
    u = -K @ state[0:6]

    # Cap control
    u_magnitude = np.linalg.norm(u)
    if(u_magnitude > u_max):
        u = (u / u_magnitude)*u_max
        
    # Calculate mass change
    isp = 3300
    delta_mass = -np.linalg.norm(u)*(state[6]/(isp*(9.80665E-3)))
    
    return u, delta_mass

def nextState(t, state, A, B, K, u_max, satellite_mass):
    # Get control
    [u, dMass] = calculateControl(state, u_max, A, B, K)

    # Calculate change
    delta_state = np.append((A @ state[0:6]) + (B @ u), dMass)
    return delta_state


## SIMULATION INTEGRATION
def simulate(
        state, 
        time_range, 
        q_weights, 
        r_weights, 
        A, 
        B,
        u_max,
        satellite_mass
        ):

    [Q, R] = matrices(q_weights, r_weights)
    [K, S, E] = ctrl.lqr(A, B, Q, R)

    sol = integrate.solve_ivp(
        nextState,
        time_range,
        state,
        # max_step = 0.25,
        args=(A, B, K, u_max, satellite_mass),
        events=(convergeEvent, massEvent),
        atol=1e-6,
        rtol=1e-3
    )

    return sol



## EVENTS
def convergeEvent(t, state, A, B, K, u_max, satellite_mass):
    pos_tol = 1e-3
    vel_tol = 1e-6

    tol = np.array([pos_tol, pos_tol, pos_tol, vel_tol, vel_tol, vel_tol])

    exit = 0
    for i in range(6):
        if abs(state[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1
convergeEvent.terminal = True
convergeEvent.direction = 0

def massEvent(t, state, A, B, K, u_max, satellite_mass):
    if state[6] < satellite_mass:
        return 0
    return 1
massEvent.terminal = True
massEvent.direction = 0 