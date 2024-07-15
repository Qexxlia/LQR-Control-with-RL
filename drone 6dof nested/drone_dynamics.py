# Imports
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy import integrate
from math import copysign

def precalcMatrices(Kt, Kf, L, Jr, Jp, Jy, m, g):
    g = 9.81

    A_ang = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
    , dtype=np.float32)

    B_ang = np.array([
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [-Kt/Jy,    -Kt/Jy,    Kt/Jy,    Kt/Jy],
            [L*Kf/Jp,    -L*Kf/Jp,    0,    0],
            [0,    0,    L*Kf/Jr,    -L*Kf/Jr],
        ]
    , dtype=np.float32)
    
    A_pos = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
    ])
    
    B_pos = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, g, 0],
        [g, 0, 0],
        [0, 0, 1/m]
    ])
    
    return A_ang, B_ang, A_pos, B_pos

def angle_controller(ref, X, A, B, K, max_attitude):
    
    for i in range(3):
        if abs(ref[i]) > max_attitude:
            ref[i] = copysign(max_attitude, ref[i])
    
    e = X - np.pad(ref, (0, len(X)-len(ref)), 'constant')
    u = -K@e
    
    dadt = A@X + B@u

    return dadt

def position_controller(_t, X, ref, A_pos, B_pos, K_pos, A_ang, B_ang, K_ang, max_attitude):
    X_pos = X[0:6]
    X_ang = X[6:12]

    e = X_pos - np.pad(ref, (0, len(X_pos)-len(ref)), 'constant')
    u_desired = -K_pos@e
    
    dadt = angle_controller(u_desired, X_ang, A_ang, B_ang, K_ang, max_attitude)
    
    u = X[7:10] + dadt[0:3]
    
    dxdt = A_pos@X_pos + B_pos@u
    
    return np.append(dxdt, dadt)

## SIMULATION INTEGRATION
def simulate(
        state, 
        time_range, 
        ref,
        A_pos, B_pos, Q_pos, R_pos,
        A_ang, B_ang, Q_ang, R_ang,
        max_attitude
        ):
    
    (K_pos, _, _) = ctrl.lqr(A_pos, B_pos, Q_pos, R_pos)
    (K_ang, _, _) = ctrl.lqr(A_ang, B_ang, Q_ang, R_ang)
    
    sol = integrate.solve_ivp(
        position_controller,
        time_range,
        state,
        args=(ref, A_pos, B_pos, K_pos, A_ang, B_ang, K_ang, max_attitude),
        events=(convergeEvent),
        atol=1e-6,
        rtol=1e-3
    )

    return sol


## EVENTS
def convergeEvent(t, X, ref, A_pos, B_pos, K_pos, A_ang, B_ang, K_ang, max_attitude):

    tol = np.array([1e-3, 1e-3, 1e-3])

    exit = 0
    for i in range(3):
        if abs(X[i] - ref[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1
convergeEvent.terminal = True
convergeEvent.direction = 0