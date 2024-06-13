import gymnasium as gym
import numpy as np
import pandas as pd
import math

import SpacecraftDynamics as scd

class SpacecraftEnv(gym.Env):
    
    def __init__(self, verbose, tStep, variance_percentage, maxDuration, map_limits, u_max):
        # Copy parameters
        self.tStep = tStep
        self.variance_percentage = variance_percentage
        self.maxDuration = maxDuration
        self.map_limits = map_limits
        self.u_max = u_max
        self.verbose = verbose

        # Define the inital state
        self.initialState = np.array([
            0.5,    # x
            -0.5,    # y
            0,    # z
            1e-3,   # x_dot
            -1e-3,   # y_dot
            0,   # z_dot
            30,   # mass 
        ], dtype=np.float32)

        self.A = scd.calcAMatrix(6371 + 500, 3.986e5)
        
        self.maxPos = max(abs(self.initialState[0:3])) 
        self.maxVel = self.maxPos/10

        # Define the action space
        actionLimits = np.ones(9, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low = -actionLimits,
            high = actionLimits,
            dtype = np.float32
        )
        
        # Define the observation space
        observationLimits = np.ones(7, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low = 0*observationLimits,
            high = observationLimits,
            dtype = np.float32
        )

        # Define timeframe
        self.currentTime = 0
        self.t0 = 0
        
        # Define deltaV usage
        self.dVT = 0
        
        # Tracking Time / Weight Updates
        self.numUpdates = 0
        
        # Tracking position
        self.pos = np.atleast_2d(self.initialState[0:3]).T
        self.vel = np.atleast_2d(self.initialState[3:6]).T
        
        np.random.seed(0)
        
    def step(self, action):
        # Define action
        qWeights = self.map_range(action[0:6], -1, 1, self.map_limits[0,0:6], self.map_limits[1,0:6])
        rWeights = self.map_range(action[6:9], -1, 1, self.map_limits[0,6:9], self.map_limits[1,6:9])
        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("Action:", action)
                print("Q:", qWeights)
                print("R:", rWeights)
        
        timeRange = (self.currentTime, self.currentTime + self.tStep)
     
        # Simulate dynamics
        sol = scd.simulate(self.state, timeRange, qWeights, rWeights, self.A, self.u_max)
        
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
        if self.currentTime >= self.maxDuration:
            truncated = True
        
        # Calculate reward
        reward = 0
        timePunishment = self.currentTime - self.t0
        deltaVPunishment = self.dVT
        distancePunishment = np.linalg.norm(sol.y[0:3, -1])
        velocityPunishment = np.linalg.norm(sol.y[3:6, -1])
        truncatedPunishment = 0

        if noDeltaV:
            deltaVPunishment = 250
        if truncated:
            truncatedPunishment = 250
        
        reward = - timePunishment - deltaVPunishment - distancePunishment - velocityPunishment - truncatedPunishment
            
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
        # self.var_state()
        self.range_state()
        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("State: ", self.state)

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
        
        return abs(normalisedState)


    def map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min
    
    def var_state(self):
        r = np.append(np.random.normal(-self.variance_percentage, self.variance_percentage, 6), 0)
        self.state = self.initialState*(1+r)
    
    def range_state(self):
        r = math.sqrt(self.initialState[0]**2 + self.initialState[1]**2 + self.initialState[2]**2)
        theta = np.random.uniform(0, np.pi/2)
        phi = np.random.uniform(0, np.pi/2)

        v = math.sqrt(self.initialState[3]**2 + self.initialState[4]**2 + self.initialState[5]**2)
        gamma = np.random.uniform(0, np.pi/2)
        alpha = np.random.uniform(0, np.pi/2)

        self.state = np.array([0, 0, 0, 0, 0, 0, self.initialState[6]], dtype=np.float32)
        self.state[0] = r * math.sin(theta) * math.cos(phi)
        self.state[1] = r * math.sin(theta) * math.sin(phi)
        self.state[2] = r * math.cos(theta)

        self.state[3] = v * math.sin(gamma) * math.cos(alpha)
        self.state[4] = v * math.sin(gamma) * math.sin(alpha)
        self.state[5] = v * math.cos(gamma)
        

