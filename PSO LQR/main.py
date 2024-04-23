import scipy as sp
import numpy as np
import pyswarms as ps

import SpacecraftDynamics as scd
import DroneDynamics as dd

''' SPACECRAFT DYNAMICS '''

def spacecraft():
    print('Spacecraft Dynamics')
    
    # Constants
    a = 7500
    mu = 3.986e5
    n = np.sqrt(mu/a**3)

    # Inital State
    state = np.array([
        8.205e-2,
        0.816,
        -3.056e-3,
        -1.014e-4,
        -1.912e-4,
        9.993e-4,
    ], dtype=np.float32)

    # Weights
    qWeights = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    rWeights = np.array([1, 1, 1], dtype=np.float32)
    
    # Tolerance
    tol = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    # Solve
    sol = scd.simulate(state, 0, 20, n, qWeights, rWeights, tol)

    # Plot
    scd.plot(sol)


def spacecraft_pso():
    print('Spacecraft Dynamics PSO')
    
    # Constants
    a = 7500
    mu = 3.986e5
    n = np.sqrt(mu/a**3)
    
    # Initial State
    state = np.array([
        8.205e-2,
        0.816,
        -3.056e-3,
        -1.014e-4,
        -1.912e-4,
        9.993e-4,
    ], dtype=np.float32)
    
    # Weight Matrices
    rWeights = np.array([1, 1, 1])

    # Tolerance
    tol = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    
    # Hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    # Optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=6, options=options)
    
    kwargs = {"rWeights":rWeights, "state":state, "t0":0, "tf":20, "n":n, "tol":tol}
    cost, pos = optimizer.optimize(scd.psoSimulateQ, 1000, **kwargs)
    
    

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