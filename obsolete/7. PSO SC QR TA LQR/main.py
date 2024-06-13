import scipy as sp
import numpy as np
import pyswarms as ps
import control

import SpacecraftDynamics as scd
# import DroneDynamics as dd

import PSOEnv as env

''' SPACECRAFT DYNAMICS '''

def spacecraft():
    print('Spacecraft Dynamics')
    
    # Initial State
    state = np.array([
        1,
        1,
        1,
        0,
        0,
        0,
        1000,
    ], dtype=np.float32)

    # Weights
    qWeights = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    
    # Solve
    sol = scd.simulate(state, (0, 200000000), qWeights)

    # Plot
    print(sol.y.shape)
    scd.plot(sol)


def spacecraft_pso():
    print('Spacecraft Dynamics PSO')
    
    # Hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    # Bounds
    bounds = (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]))
    
    # Optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=225, dimensions=9, options=options, bounds=bounds)

    cost, pos = optimizer.optimize(env.simulate, 2500)
    
# spacecraft_pso()

# env.simulate(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]]))

A = scd.calcAMatrix(6371 + 500, 3.986e5)
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

Q = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([1, 1, 1])

[k, a, c] = control.lqr(A, B, Q, R, method='scipy')

print(k, a, c)