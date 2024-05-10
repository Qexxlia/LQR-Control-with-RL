import SpacecraftDynamics as scd
import numpy as np

def simulate(qWeights):
    timeRange = (0, 200000)
    t0 = timeRange[0]

    state = np.array([
        1,    # x
        1,    # y
        1,    # z
        1e-4,   # x_dot
        1e-4,   # y_dot
        1e-4,   # z_dot
        1000,   # mass 
    ], dtype=np.float32)
    
    size = np.shape(qWeights)[0]
    reward = np.zeros(size)
    
    # Simulate dynamics

    for i in range(0, size):
        sol = scd.simulate(state, timeRange, qWeights[i])
        
        dVT = 0
        dVT -= sol.y[6, -1] - sol.y[6, 0]
        totalTime = sol.t[-1]

        # Check if converged
        noDeltaV = False

        if sol.status == 1:
            if sol.t_events[1].size != 0:
                noDeltaV = True
        
        # Calculate reward
        timePunishment = totalTime - t0
        deltaVPun = dVT
        
        if noDeltaV:
            reward[i]= -(-5000)
        else:
            reward[i] = -(- timePunishment - deltaVPun)

    # Return
    # print("DeltaV: ", dVT)
    # print("Time: ", totalTime)
    # print("Reward: ", reward[i])

    return reward