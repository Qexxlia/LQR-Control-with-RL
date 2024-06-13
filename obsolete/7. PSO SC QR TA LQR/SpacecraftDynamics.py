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

## CALCULATIONS
def calcAMatrix(a, mu):
    n = np.sqrt(mu/a**3)

    A = np.array(
        [
            [  0,   0,   0,   1,   0,   0],
            [  0,   0,   0,   0,   1,   0],
            [  0,   0,   0,   0,   0,   1],
            [3*n**2,   0,   0,   0, 2*n,   0],
            [  0,   0,   0, -2*n,   0,   0],
            [  0,   0, -n**2,   0,   0,   0]
        ], dtype=np.float32
    )
    
    return A

def matrices(qWeights, rWeights):
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

    Q = np.diag(qWeights)

    R = np.diag(rWeights)

    return B, Q, R

def calculateControl(state, u_max, A, B, Q, R):
    # Solve LQR for K using control library
    print(A)
    print(B)
    print(Q)
    print(R)
    [K, S, E] = ctrl.lqr(A, B, Q, R, method='scipy')

    # Calculate control
    u = -K @ state[0:6]

    # Cap control
    u_max = u_max # CANNOT BE BELOW 4.1e-6
    normU = np.linalg.norm(u)
    if(normU > u_max):
        u = (u / normU)*u_max
        
    # Calculate mass change
    Isp = 3300
    dMass = -np.linalg.norm(u)*(state[6]/(Isp*(9.80665E-3)))
    
    return u, dMass

def nextState(t, state, qWeights, rWeights, A, u_max):
    # Get control
    [B, Q, R] = matrices(qWeights, rWeights)
    [u, dMass] = calculateControl(state, u_max, A, B, Q, R)

    # Calculate change
    dState = np.append((A @ state[0:6]) + (B @ u), dMass)
    return dState




## SIMULATION INTEGRATION
def simulate(state, timeRange, qWeights, rWeights, A, u_max):
    sol = integrate.solve_ivp(
        nextState,
        timeRange,
        state,
        # max_step = 0.25,
        args=(qWeights, rWeights, A, u_max),
        events=(convergeEvent, massEvent),
        atol=1e-6,
        rtol=1e-3
    )

    return sol



## EVENTS
def convergeEvent(t, state, qWeights, rWeights, A, u_max):
    posTol = 1e-3
    velTol = 1e-6

    tol = np.array([posTol, posTol, posTol, velTol, velTol, velTol])

    exit = 0
    for i in range(5):
        if abs(state[i]) > tol[i]:
            exit += 1

    if exit == 0:
        return 0

    return 1
convergeEvent.terminal = True
convergeEvent.direction = 0

def massEvent(t, state, qWeights, rWeights, A, u_max):
    satelliteMass = 15 
    if state[6] < satelliteMass:
        return 0
    return 1
massEvent.terminal = True
massEvent.direction = 0 




## PLOTTING
def plot1(pos, vel, t):
    # Colors
    xc = 'red'
    yc = 'blue'
    zc = 'green'

    # Lineweight
    lw = 1

    # Text Sizes
    tsL = 12
    tsS = 7

    [fig, ax] = plt.subplots(2, sharex=True, figsize=(10, 5))
    fig.suptitle('State vs Time', fontsize=tsL)
    ax[1].set_xlabel('Time (s)', fontsize=tsS)
    
    ax[0].plot(t, pos[0, :], label='x', color=xc, linewidth=lw)
    ax[0].plot(t, pos[1, :], label='y', color=yc, linewidth=lw)
    ax[0].plot(t, pos[2, :], label='z', color=zc, linewidth=lw)

    ax[0].set_title('Position vs Time', fontsize=tsL)
    ax[0].set_ylabel('Position (km)', fontsize=tsS)

    ax[1].plot(t, vel[0, :], color=xc, linewidth=lw)
    ax[1].plot(t, vel[1, :], color=yc, linewidth=lw)
    ax[1].plot(t, vel[2, :], color=zc, linewidth=lw)

    ax[1].set_title('Velocity vs Time', fontsize=tsL)
    ax[1].set_ylabel('Velocity (km/s)', fontsize=tsS)

    fig.legend(loc='upper right')
    
    return fig

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
    ax[0].set_ylabel('Position (km)', fontsize=tsS)

    ax[1].plot(sol.t, sol.y[3, :], color=xc, linewidth=lw)
    ax[1].plot(sol.t, sol.y[4, :], color=yc, linewidth=lw)
    ax[1].plot(sol.t, sol.y[5, :], color=zc, linewidth=lw)

    ax[1].set_title('Velocity vs Time', fontsize=tsL)
    ax[1].set_ylabel('Velocity (km/s)', fontsize=tsS)

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
