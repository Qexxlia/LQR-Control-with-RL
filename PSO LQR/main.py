import scipy as sp
import numpy as np
import pyswarms as ps

import SpacecraftDynamics as scd
import DroneDynamics as dd

import PSOEnv as env

''' SPACECRAFT DYNAMICS '''

def spacecraft():
    print('Spacecraft Dynamics')
    
    # Initial State
    state = np.array([
        1,
        -1,
        1.1,
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
    bounds = (np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1]))
    
    # Optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=6, options=options, bounds=bounds)

    cost, pos = optimizer.optimize(env.simulate, 100)
    

''' DRONE DYNAMICS '''


def drone():
    print('Drone Dynamics')

    # Constants
    m = 1
    Ix = 0.11
    Iy = 0.11
    Iz = 0.04

    # Initial State
    state = np.array([
        2,
        3,
        15,
        15,
        10,
        3,
        0,
        0,
        0,
        0,
        0,
        0
    ], dtype=np.float32)

    # Weight Matrices
    qWeights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    rWeights = np.array([1, 1, 1, 1], dtype=np.float32)

    # Tolerance
    tol = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    sol = dd.simulate(state, 0, 10, m, Ix, Iy, Iz, qWeights, rWeights, tol)

    dd.plot(sol)
    
spacecraft_pso()