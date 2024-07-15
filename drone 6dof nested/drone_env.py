import gymnasium as gym
import numpy as np
import pandas as pd
import math
from scipy import integrate

import drone_dynamics as dd

class DroneEnv(gym.Env):
    
    def __init__(self, verbose, args = None):
        # Copy parameters
        self.t_step_limits = args.get('t_step_limits')
        self.variance = args.get('variance')
        self.variance_type = args.get('variance_type')
        self.max_duration = args.get('max_duration')
        self.map_limits = args.get('map_limits')
        self.u_max = args.get('u_max')
        self.seed = args.get('seed')
        self.absolute_norm = args.get('absolute_norm')
        self.verbose = verbose
        
        self.ref = np.array([1, -1, 0.5], dtype=np.float32)

        # Define the initial state
        self.initial_state = np.zeros(12)
        
        # Calculate A matrix for the dynamics
        (self.A_ang, self.B_ang, self.A_pos, self.B_pos) = dd.precalcMatrices(0.0036, 0.1188, 0.1969, 0.0552, 0.0552, 0.1104, 1, 9.81)
        
        # Calculate the maximum position and velocity for normalisation
        self.max_pos = 1.1*self.ref
        self.max_attitude = 0.01

        # Define the action space
        action_limits = np.ones(14, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low = -action_limits,
            high = action_limits,
            dtype = np.float32
        )
        
        # Change the observation space if absolute normalisation is used
        if(self.absolute_norm):
            boolabs = 0
        else:
            boolabs = -1
        
        # Define the observation space
        observation_limits = np.ones(6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low = boolabs*observation_limits,
            high = observation_limits,
            dtype = np.float32
        )

        # Define timeframe
        self.current_time = 0
        self.initial_time = 0
        self.t_step = self.t_step_limits[0]
        
        # Tracking position
        self.position = np.atleast_2d(self.initial_state[0:3]).T
        self.attitude = np.atleast_2d(self.initial_state[7:10]).T
        
        # Set the random seed
        np.random.seed(self.seed)
        
        # Episode options
        self.episode_options = {}

    def step(self, action):
        # Define action
        Q_ang = np.diag(self.log_map_range(action[0:6], -1, 1, np.ones(6)*self.map_limits[0,0], np.ones(6)*self.map_limits[1,0]))
        R_ang = self.log_map_range(action[6], -1, 1, self.map_limits[0,1], self.map_limits[1,1])*np.eye(4)
        Q_pos = np.diag(self.log_map_range(action[7:13], -1, 1, np.ones(6)*self.map_limits[0,2], np.ones(6)*self.map_limits[1,2]))
        R_pos = self.log_map_range(action[13], -1, 1, self.map_limits[0,3], self.map_limits[1,3])*np.eye(3)

        # Set time range
        time_range = (self.current_time, self.current_time + self.t_step)
     
        # Simulate dynamics
        if self.verbose == 1: print("Simulating...")
        sol = dd.simulate(self.state, time_range, self.ref, self.A_pos, self.B_pos, Q_pos, R_pos, self.A_ang, self.B_ang, Q_ang, R_ang, self.max_attitude)
        if self.verbose == 1: print("Simulated")
        
        # Update state
        self.current_time = sol.t[-1]
        self.state = sol.y[:,-1]

        self.attitude = np.hstack((np.atleast_2d(self.attitude), sol.y[7:10,:]))
        self.position = np.hstack((np.atleast_2d(self.position), sol.y[0:3,:]))
        self.t = np.hstack((self.t, sol.t))
       
        normalised_state = self.normalise_state(self.state)

        terminated = False
        truncated = False

        if sol.t_events[0].size != 0:
            terminated = True
        if self.current_time >= self.max_duration:
            truncated = True

        reward = 0

        #$$ Reward Calculation
        if terminated:
            reward = -self.current_time
        if truncated:
            reward += -np.linalg.norm(self.position[:,-1] - self.ref)
            
        #$$
        
        # Return
        info = {
                'attitude': self.attitude,
                'position': self.position,
                't' : self.t,
                }
            
        return normalised_state, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = {}):
        # Reset state
        options = self.episode_options
        
        # Allow single verbose run
        if(options.get("verbose", 0) == 1):
            self.verbose = 1
        else:
            self.verbose = 0

        self.deterministic_state()

        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("State: ", self.state)

        # Reset state vars
        self.current_time = 0
        self.num_updates = 0
        self.initial_time = 0
        self.t = np.zeros(1)

        self.position = np.atleast_2d(self.initial_state[0:3]).T
        self.attitude = np.atleast_2d(self.initial_state[7:10]).T
       
        self.episode_options = {}
        
        normalised_state = self.normalise_state(self.state)

        return normalised_state, {}
    

    def normalise_state (self, state):
        # Normalise the state
        normalised_state = np.hstack((state[0:3]/self.max_pos, state[7:10]/self.max_attitude))
        
        if self.absolute_norm:
            normalised_state = np.abs(normalised_state)
        
        return normalised_state


    def linear_map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min # LINEAR
    
    def log_map_range(self, val, in_min, in_max, out_min, out_max):
        return out_min * (out_max/out_min)**((val - in_min)/(in_max - in_min)) # LOG
    
    def deterministic_state(self):
        self.state = self.initial_state
        
    def set_episode_options(self, options):
        self.episode_options = options
        
    def update_u_max(self, u_max):
        self.u_max = u_max