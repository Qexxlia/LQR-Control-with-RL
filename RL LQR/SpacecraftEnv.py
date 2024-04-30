import gymnasium as gym
import numpy as np

import SpacecraftDynamics as scd

class SpacecraftEnv(gym.Env):
    
    def __init__(self, render_mode = "none"):

        # Copy parameters
        self.render_mode = render_mode

        # Define the inital state
        self.initialState = np.array([
            8.205e-2,   # x
            0.816,      # y
            -3.056e-3,  # z
            -1.014e-4,  # x_dot
            -1.912e-4,  # y_dot
            9.993e-4,   # z_dot
            1000,       # mass 
        ], dtype=np.float32)
        

        # Define the action space
        actionLimits = np.array([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low = -actionLimits,
            high = actionLimits,
            dtype = np.float32
        )
        
        # Define the observation space
        
        observationLimits = np.array([
            100000,
            100000,
            100000,
            1000,
            1000,
            1000,
            1000,
        ], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low = -observationLimits,
            high = observationLimits,
            dtype = np.float32
        )

        # Define timeframe
        self.t0 = 0
        self.tf = 20000000 
        
        self.dVT = 0
        
        # Tracking Time / Weight Updates
        self.totalTime = 0
        self.numUpdates = 0
        
    def step(self, action):
        # Define action
        qWeights = action
        
        timeRange = (self.t0, self.tf)
     
        # Simulate dynamics
        sol = scd.simulate(self.state, timeRange, qWeights)
        
        self.numUpdates += 1
        self.totalTime += sol.t[-1]
        
        self.dVT -= sol.y[6, -1] - sol.y[6, 0]

        # Check if converged
        terminated = False
        noDeltaV = False

        if sol.status == 1:
            if sol.t_events[0].size != 0:
                converged = True
                terminated = True
            elif sol.t_events[1].size != 0:
                noDeltaV = True
                terminated = True
        else:
            self.state = sol.y[:,-1]
            
        # Check if truncated #TODO
        truncated = False
            
        # Calculate reward
        timePunishment = (self.totalTime - self.t0)*0.25
        deltaVPun = self.dVT
        
        print(timePunishment)
        print(deltaVPun)
            
        if noDeltaV:
            reward = -5000
        else:
            reward = -timePunishment + -deltaVPun
        
        print(reward)
        print()

        # Return
        return self.state, reward, terminated, truncated, {}
    
    def reset(self, *, seed = None, options = None):
        # Reset state
        self.state = self.initialState

        self.dVT = 0

        self.totalTime = 0
        self.t0 = 0
        
        self.numUpdates = 0

        return np.array(self.state), {}