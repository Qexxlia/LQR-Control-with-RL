import gymnasium as gym
import numpy as np
import pandas as pd

import SpacecraftDynamics as scd

class SpacecraftEnv(gym.Env):
    
    def __init__(self, render_mode = "none"):
        # Copy parameters
        self.render_mode = render_mode

        # Define the inital state
        self.initialState = np.array([
            1,    # x
            1,    # y
            1,    # z
            1e-4,   # x_dot
            1e-4,   # y_dot
            1e-4,   # z_dot
            1000,   # mass 
        ], dtype=np.float32)
        
        self.maxPos = 1
        self.maxVel = 0.5

        # Define the action space
        actionLimits = np.ones(3, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low = -actionLimits,
            high = actionLimits,
            dtype = np.float32
        )
        
        # Define the observation space
        observationLimits = np.ones(7, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low = -observationLimits,
            high = observationLimits,
            dtype = np.float32
        )

        # Define timeframe
        self.tStep = 10
        self.currentTime = 0
        self.t0 = 0
        
        # Define deltaV usage
        self.dVT = 0
        
        # Tracking Time / Weight Updates
        self.numUpdates = 0
        
        # Tracking position
        self.pos = np.atleast_2d(self.initialState[0:3]).T
        self.vel = np.atleast_2d(self.initialState[3:6]).T
        
    def step(self, action):
        # Define action
        qWeights = np.ones(6)
        rWeights = self.map_range(action, -1, 1, 1e-8, 1)
        
        timeRange = (self.currentTime, self.currentTime + self.tStep)
     
        # Simulate dynamics
        sol = scd.simulate(self.state, timeRange, qWeights, rWeights)
        
        self.numUpdates += 1
        self.currentTime = sol.t[-1]
        
        self.dVT -= sol.y[6, -1] - sol.y[6, 0]

        # Check if converged
        converged = False
        noDeltaV = False
        terminated = False
        truncated = False

        if sol.t_events[0].size != 0:
            converged = True
            terminated = True
        elif sol.t_events[1].size != 0:
            noDeltaV = True
            truncated = True
        
        # Calculate reward
        reward = 0

        if noDeltaV:
            reward = -5000
        elif converged:
            timePunishment = self.currentTime - self.t0
            deltaVPunishment = self.dVT
            reward = - timePunishment - deltaVPunishment
            
        # Update state
        self.state = sol.y[:,-1]
        
        self.pos = np.hstack((np.atleast_2d(self.pos), sol.y[0:3, :]))
        self.vel = np.hstack((np.atleast_2d(self.vel), sol.y[3:6, :]))
        self.t = np.hstack((self.t, sol.t))
       
        normalisedState = self.normalise_state(self.state)

        # Return
        return normalisedState, reward, terminated, truncated, {'pos': self.pos, 'vel': self.vel, 't': self.t, 'dVT': self.dVT}
    
    def reset(self, *, seed = None, options = None):
        # Reset state
        self.state = self.initialState

        self.dVT = 0

        self.currentTime = 0
        self.numUpdates = 0
        self.t0 = 0
        
        self.t = np.zeros(1)
        self.pos = np.atleast_2d(self.initialState[0:3]).T
        self.vel = np.atleast_2d(self.initialState[3:6]).T
        
        return self.state, {}
    

    def normalise_state (self, state):
        # Normalise the state
        normalisedState = np.zeros(state.shape)

        normalisedState[0] = state[0]/self.maxPos
        normalisedState[1] = state[1]/self.maxPos
        normalisedState[2] = state[2]/self.maxPos
        normalisedState[3] = state[3]/self.maxVel
        normalisedState[4] = state[4]/self.maxVel
        normalisedState[5] = state[5]/self.maxVel
        normalisedState[6] = state[6]/self.initialState[6]
        
        return normalisedState


    def map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min