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
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low = -actionLimits,
            high = actionLimits,
            dtype = np.float32
        )
        
        # Define the observation space
        
        observationLimits = np.array([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low = -observationLimits,
            high = observationLimits,
            dtype = np.float32
        )

        # Define timeframe
        self.tStep = 10
        self.currentTime = 0
        self.t0 = 0
        
        self.dVT = 0
        
        # Tracking Time / Weight Updates
        self.numUpdates = 0
        
    def step(self, action):
        # Define action
        qWeights = action
        
        timeRange = (self.currentTime, self.currentTime + self.tStep)
     
        # Simulate dynamics
        sol = scd.simulate(self.state, timeRange, qWeights)
        
        self.numUpdates += 1
        self.currentTime = sol.t[-1]
        
        self.dVT -= sol.y[6, -1] - sol.y[6, 0]

        # Check if converged
        terminated = False
        converged = False
        noDeltaV = False

        if sol.t_events[0].size != 0:
            converged = True
            terminated = True
        elif sol.t_events[1].size != 0:
            noDeltaV = True
            terminated = True

            
        # Cannot be truncated with fixed time step
        truncated = False
            
        
        # Calculate reward
        reward = 0

        if noDeltaV:
            reward = -5000
        elif converged:
            timePunishment = self.currentTime - self.t0
            deltaVPunishment = self.dVT
            reward = - timePunishment - deltaVPunishment
            # print("TP : ", timePunishment, " DVP: ", deltaVPunishment, " R: ", reward)
            
        # Update state
        self.state = sol.y[:,-1]
        
        # scd.plot(sol)

        normalisedState = self.normalise_state(self.state)

        # Return
        return normalisedState, reward, terminated, truncated, {}
    
    def reset(self, *, seed = None, options = None):
        # Reset state
        self.state = self.initialState

        self.dVT = 0

        self.currentTime = 0
        self.t0 = 0
        
        self.numUpdates = 0

        return self.state, {}
    
    def normalise_state (self, state):
        normalisedState = np.zeros(state.shape)
        
        normalisedState[0] = state[0]/self.initialState[0]
        normalisedState[1] = state[1]/self.initialState[1]
        normalisedState[2] = state[2]/self.initialState[2]
        normalisedState[3] = state[3]/self.initialState[3]
        normalisedState[4] = state[4]/self.initialState[4]
        normalisedState[5] = state[5]/self.initialState[5]
        normalisedState[6] = state[6]/self.initialState[6]
        
        return normalisedState