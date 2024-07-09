# Imports
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy import integrate

''' 
Lqr controller for the drone dynamics
Taken from Simulation of the Quadcopter Dynamics with LQR based control, Faraz Ahmad et al, 2018
'''

'''

Yaw is phi
Pitch is theta
Roll is psi

STATE VECTOR of form

[
    r, p, y, dr, dp, dy
]
'''

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
            [0, 0, 0, 0, 0, 0]
        ]
    , dtype=np.float32)

    B = np.array([
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [-Kt/Jy,    -Kt/Jy,    Kt/Jy,    Kt/Jy],
            [L*Kf/Jp,    -L*Kf/Jp,    0,    0],
            [0,    0,    L*Kf/Jr,    -L*Kf/Jr],
        ]
    , dtype=np.float32)
    
    return A, B

class DroneDynamics:
    
    prev_u = np.zeros(4, dtype=np.float32)

    def calculateControl(self, state, u_max, K):
        
        # Calculate control
        u = -K @ state

        # Cap control
        ex = False
        for ctrl in u:
            if abs(ctrl) > u_max:
                ex = True
                break
        if ex:
            u = u / max(abs(u)) * u_max
        return u

    def nextState(self, t, state, q_weights, r_weights, A, B, K, u_max):
        # Get control
        state = state[0:6]

        u = self.calculateControl(state, u_max, K)
        
        # Calculate change
        delta_state = (A @ state) + (B @ u)
        
        delta_state = np.hstack((delta_state, u - self.prev_u))
        
        self.prev_u = u

        return delta_state

    ## SIMULATION INTEGRATION
def simulate(
        state, 
        time_range, 
        q_weights, 
        r_weights, 
        A, 
        B,
        u_max
        ):
    
    Q = np.diag(q_weights)
    R = np.diag(r_weights)
    
    [K, S, E] = ctrl.lqr(A, B, Q, R)
    
    state = np.hstack((state, np.zeros(4)))

    dyn = DroneDynamics()

    sol = integrate.solve_ivp(
        dyn.nextState,
        time_range,
        state,
        # max_step = 0.25,
        args=(q_weights, r_weights, A, B, K, u_max),
        events=(convergeEvent),
        atol=1e-6,
        rtol=1e-3
    )
    
    u = sol.y[6:,:]
    sol.y = sol.y[0:6,:]
    return sol, u


## EVENTS
def convergeEvent(t, state, q_weights, r_weights, A, B, K, u_max):
    attitude_tol = 0.005
    spin_tol = 0.0005

    tol = np.array([attitude_tol, attitude_tol, attitude_tol, spin_tol, spin_tol, spin_tol])

    exit = 0
    for i in range(6):
        if abs(state[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1
convergeEvent.terminal = True
convergeEvent.direction = 0