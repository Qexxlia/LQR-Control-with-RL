import SpacecraftDynamics as scd
import numpy as np

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def simulate(action):
    time_range = (0, 10000)
    t0 = time_range[0]
    
    u_max = 1e-3
    
    A = scd.calcAMatrix(6371 + 500, 3.986e5)

    state = np.array([
        0.5,    # x
        -0.5,    # y
        0,    # z
        1e-3,   # x_dot
        -1e-3,   # y_dot
        0,   # z_dot
        30,   # mass 
    ], dtype=np.float32)
    
    size = np.shape(action)[0]
    reward = np.zeros(size)
    
    # Simulate dynamics

    for i in range(0, size):

        sol = scd.simulate(state, time_range, action[i,0:6], action[i,6:9], A, u_max)
        
        dVT = 0
        dVT -= sol.y[6, -1] - sol.y[6, 0]
        totalTime = sol.t[-1]

        # Check if converged
        converged = False
        no_deltaV = False
        terminated = False
        truncated = False

        if sol.t_events[0].size != 0:
            converged = True
            terminated = True
        elif sol.t_events[1].size != 0:
            no_deltaV = True
            truncated = True
            
        # Calculate reward
        reward = 0
        time_punishment = totalTime
        deltaV_punishment = dVT * 200
        distance_punishment = np.linalg.norm(sol.y[0:3, -1])
        velocity_punishment = np.linalg.norm(sol.y[3:6, -1])
        truncated_punishment = 0
        
        if no_deltaV:
            deltaV_punishment = 250
        if truncated:
            truncated_punishment = 250
            
            reward = -time_punishment - deltaV_punishment - distance_punishment - velocity_punishment - truncated_punishment
            reward = -reward
    return reward