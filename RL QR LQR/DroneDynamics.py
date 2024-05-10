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
'''




import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy import integrate
def matrices(m, Ix, Iy, Iz, qWeights, rWeights):
    g = 9.81

    A = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )

    mc = -1/m
    Ixc = 1/Ix
    Iyc = 1/Iy
    Izc = 1/Iz

    B = np.array(
        [
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [0,  Ixc,    0,    0],
            [0,    0,  Iyc,    0],
            [0,    0,    0,  Izc],
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [mc,    0,    0,    0],
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [0,    0,    0,    0]
        ]
    )

    Q = np.diag(qWeights)
    R = np.diag(rWeights)

    return A, B, Q, R


def calculateControl(state, A, B, Q, R):

    # Solve LQR for K using control library
    [K, S, E] = ctrl.lqr(A, B, Q, R, method='scipy')

    # Calculate control
    u = np.matmul(-K, state)

    # Cap control
    # u_max = 1e6

    # if(np.linalg.norm(u) > u_max):
    # u = (u / np.linalg.norm(u))*u_max

    return u


def nextState(t, state, m, Ix, Iy, Iz, qWeights, rWeights, tol):

    # Get control
    [A, B, Q, R] = matrices(m, Ix, Iy, Iz, qWeights, rWeights)
    u = calculateControl(state, A, B, Q, R)

    # Calculate change
    dState = np.matmul(A, state) + np.matmul(B, u)

    return dState


def simulate(state, t0, tf, m, Ix, Iy, Iz, qWeights, rWeights, tol):

    # Integrate
    sol = integrate.solve_ivp(
        nextState,
        (t0, tf),
        state,
        args=(m, Ix, Iy, Iz, qWeights, rWeights, tol),
        events=convergeEvent
    )

    return sol


def convergeEvent(t, state, m, Ix, Iy, Iz, qWeights, rWeights, tol):
    exit = 0
    for i in range(12):
        if abs(state[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1


convergeEvent.terminal = True
convergeEvent.direction = 0


def plot(sol):
    # Colors
    xc = 'red'
    yc = 'blue'
    zc = 'green'
    psic = 'orange'
    thetac = 'aquamarine'
    phic = 'violet'

    # Lineweight
    lw = 0.5

    # Text Sizes
    tsL = 12
    tsS = 7

    [fig, ax] = plt.subplots(4, sharex=True)
    fig.suptitle('State vs Time', fontsize=tsL)
    ax[3].set_xlabel('Time (s)', fontsize=tsS)

    # Plot Position
    ax[0].plot(sol.t, sol.y[0, :], label='x', color=xc, linewidth=lw)
    ax[0].plot(sol.t, sol.y[1, :], label='y', color=yc, linewidth=lw)
    ax[0].plot(sol.t, sol.y[2, :], label='z', color=zc, linewidth=lw)

    # Plot Angles
    ax[1].plot(sol.t, sol.y[3, :], label='psi', color=psic, linewidth=lw)
    ax[1].plot(sol.t, sol.y[4, :], label='theta', color=thetac, linewidth=lw)
    ax[1].plot(sol.t, sol.y[5, :], label='phi', color=phic, linewidth=lw)

    ax[1].set_ylabel('Angle (deg)', fontsize=tsS)

    # Plot Velocity
    ax[2].plot(sol.t, sol.y[6, :], color=xc, linewidth=lw)
    ax[2].plot(sol.t, sol.y[7, :], color=yc, linewidth=lw)
    ax[2].plot(sol.t, sol.y[8, :], color=zc, linewidth=lw)

    ax[2].set_ylabel('Velocity (m/s)', fontsize=tsS)

    # Plot Angular Velocity
    ax[3].plot(sol.t, sol.y[9, :], color=psic, linewidth=lw)
    ax[3].plot(sol.t, sol.y[10, :], color=thetac, linewidth=lw)
    ax[3].plot(sol.t, sol.y[11, :], color=phic, linewidth=lw)

    ax[3].set_ylabel('Angular Velocity (deg/s)', fontsize=tsS)

    fig.legend(loc='upper right')
    plt.show()


def plot3d(sol):
    # Color
    c = 'blue'

    # Lineweight
    lw = 1

    # Text Sizes
    tsL = 12
    tsS = 7

    # Plot 3d Position
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :],
              label='Position', color=c, linewidth=lw)

    plt.title('Position', fontsize=tsL)
    ax3d.set_xlabel('x', fontsize=tsS)
    ax3d.set_ylabel('y', fontsize=tsS)
    ax3d.set_zlabel('z', fontsize=tsS)
    plt.show()
