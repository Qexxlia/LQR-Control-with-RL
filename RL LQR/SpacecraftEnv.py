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
        ], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low = -observationLimits,
            high = observationLimits,
            dtype = np.float32
        )
            
        # Constants
        mu = 3.986e5
        a = 7500
        self.n = np.sqrt(mu/a**3)

        # Define timeframe
        self.t0 = 0
        self.tf = 200
        
        # Define tolerances
        self.tol = np.array([
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
        ])
        
    def step(self, action):
        self.prevState = self.state
        
        # Define action
        qWeights = action
        rWeights = np.array([1, 1, 1], dtype=np.float32)
        
        # Simulate dynamics
        sol = scd.simulate(self.state, self.t0, self.tf, self.n, qWeights, rWeights, self.tol)
        
        # Check if converged
        terminated = False
        if sol.status == 1:
            if sol.t_events == []:
                self.state = sol.y[:,-1]
            elif "convergeEvent" in sol.t_events:
                print("ENDED")
                print(sol.t_events)
                terminated = True
            
        # Check if truncated #TODO
        truncated = False
            
        # Calculate reward
        reward = np.linalg.norm(self.state - self.prevState)

        # Return
        return self.state, reward, terminated, truncated, {}
    
    def reset(self, *, seed = None, options = None):
        
        # Reset state
        self.state = self.initialState
        
        return np.array(self.state), {}