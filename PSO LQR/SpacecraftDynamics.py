# Imports
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy import integrate

'''
State From
X = [
    x,
    y,
    z,
    x_dot,
    y_dot,
    z_dot
]
'''


def matrices(n, qWeights, rWeights):
    # Get matrices

    # Calculate A matrix
    A14 = 3*np.float_power(n, 2)
    A54 = 2*n
    A45 = -2*n
    A36 = -np.float_power(n, 2)

    # Define A,B,Q,R matrices
    A = np.array(
        [
            [  0,   0,   0,   1,   0,   0],
            [  0,   0,   0,   0,   1,   0],
            [  0,   0,   0,   0,   0,   1],
            [A14,   0,   0,   0, A54,   0],
            [  0,   0,   0, A45,   0,   0],
            [  0,   0, A36,   0,   0,   0]
        ], dtype=np.float32
    )

    B = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32
    )

    q = createDiag(qWeights)
    Q = np.dot(q, np.transpose(q))

    r = createDiag(rWeights)
    R = np.dot(r, np.transpose(r))

    return A, B, Q, R

def createDiag(weights):
    if weights.ndim == 1:
        return np.diag(weights)
    else:
        return np.diag(weights)


def calculateControl(state, A, B, Q, R):

    # Solve LQR for K using control library
    [K, S, E] = ctrl.lqr(A, B, Q, R, method='scipy')

    # Calculate control
    u = np.matmul(-K, state)

    # Cap control
    # u_max = 16
    # if(np.linalg.norm(u) > u_max):
    # u = (u / np.linalg.norm(u))*u_max

    return u


def nextState(t, state, n, qWeights, rWeights, tol):
    # Get control
    [A, B, Q, R] = matrices(n, qWeights, rWeights)
    u = calculateControl(state, A, B, Q, R)

    # Calculate change
    dState = np.matmul(A, state) + np.matmul(B, u)

    return dState


def simulate(state, t0, tf, n, qWeights, rWeights, tol):

    sol = integrate.solve_ivp(
        nextState,
        (t0, tf),
        state,
        args=(n, qWeights, rWeights, tol),
        events=convergeEvent
    )

    return sol

def calcReward(sol):
    # Calculate reward
    reward = np.linalg.norm(sol.y[0:3, -1])
    return reward

def psoSimulateQ(qWeights, rWeights, state, t0, tf, n, tol):
    reward = []
    for i in range(len(qWeights)):
        sol = simulate(state, t0, tf, n, qWeights[i], rWeights, tol)
        reward.append(calcReward(sol))
    return reward


def psoSimulateR(rWeights, qWeights, state, t0, tf, n, tol):
    sol = simulate(state, t0, tf, n, qWeights, rWeights, tol)
    return sol


def psoSimulateQR(weights, state, t0, tf, n, tol):
    qWeights = weights[0:6]
    rWeights = weights[6:9]

    sol = simulate(state, t0, tf, n, qWeights, rWeights, tol)
    return sol


def convergeEvent(t, state, n, qWeights, rWeights, tol):
    exit = 0
    for i in range(6):
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

    # Lineweight
    lw = 1

    # Text Sizes
    tsL = 12
    tsS = 7

    [fig, ax] = plt.subplots(2, sharex=True)
    fig.suptitle('State vs Time', fontsize=tsL)
    ax[1].set_xlabel('Time (s)', fontsize=tsS)

    ax[0].plot(sol.t, sol.y[0, :], label='x', color=xc, linewidth=lw)
    ax[0].plot(sol.t, sol.y[1, :], label='y', color=yc, linewidth=lw)
    ax[0].plot(sol.t, sol.y[2, :], label='z', color=zc, linewidth=lw)

    ax[0].set_title('Position vs Time', fontsize=tsL)
    ax[0].set_ylabel('Position (m/s)', fontsize=tsS)

    ax[1].plot(sol.t, sol.y[3, :], color=xc, linewidth=lw)
    ax[1].plot(sol.t, sol.y[4, :], color=yc, linewidth=lw)
    ax[1].plot(sol.t, sol.y[5, :], color=zc, linewidth=lw)

    ax[1].set_title('Velocity vs Time', fontsize=tsL)
    ax[1].set_ylabel('Velocity (m/s)', fontsize=tsS)

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

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :], linewidth=lw, color=c)
    ax3d.set_xlabel('X', fontsize=tsS)
    ax3d.set_ylabel('Y', fontsize=tsS)
    ax3d.set_zlabel('Z', fontsize=tsS)
    plt.title('Position', fontsize=tsL)
    plt.show()
