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
        
        self.desired_angle = 0.0872664625997165

        # Define the initial state
        self.initial_state = np.array([
            -self.desired_angle, # r
            -self.desired_angle, # p
            -self.desired_angle, # y
            0, # dr
            0, # dp
            0, # dy
        ], dtype=np.float32)
        
        # Calculate A matrix for the dynamics
        (self.A, self.B) = dd.precalcMatrices(0.0036, 0.1188, 0.1969, 0.0552, 0.0552, 0.1104)
        
        # Calculate the maximum position and velocity for normalisation
        self.max_attitude = 2*self.desired_angle
        self.max_spin = 6*self.desired_angle

        # Define the action space
        action_limits = np.ones(52, dtype=np.float32)

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
        self.attitude = np.atleast_2d(self.initial_state[0:3] + self.desired_angle)
        self.spin = np.atleast_2d(self.initial_state[3:6]).T
        self.u = np.atleast_2d(np.zeros(4, dtype=np.float32)).T
        
        # Set the random seed
        np.random.seed(self.seed)
        
        # Episode options
        self.episode_options = {}

    def step(self, action):
        # Define action
        # q = self.log_map_range(action[0:21], -1, 1, np.ones(21)*self.map_limits[0,0], np.ones(21)*self.map_limits[1,0])
        # r = self.log_map_range(action[21:31], -1, 1, np.ones(10)*self.map_limits[0,1], np.ones(10)*self.map_limits[1,1])
        
        # Q = np.array([
        #     [q[0], q[1],  q[2],  q[3],  q[4],  q[5]],
        #     [q[1], q[6],  q[7],  q[8],  q[9],  q[10]],
        #     [q[2], q[7],  q[11], q[12], q[13], q[14]],
        #     [q[3], q[8],  q[12], q[15], q[16], q[17]],
        #     [q[4], q[9],  q[13], q[16], q[18], q[19]],
        #     [q[5], q[10], q[14], q[17], q[19], q[20]]
        # ])
        
        # input(Q)
        
        # R = np.array([
        #     [r[0], r[1], r[2], r[3]],
        #     [r[1], r[4], r[5], r[6]],
        #     [r[2], r[5], r[7], r[8]],
        #     [r[3], r[6], r[8], r[9]]
        # ])
        q = np.reshape(self.linear_map_range(action[0:36], -1, 1, np.ones(36)*self.map_limits[0,0], np.ones(36)*self.map_limits[1,0]),(6,6))
        r = np.reshape(self.linear_map_range(action[36:52], -1, 1, np.ones(16)*self.map_limits[0,1], np.ones(16)*self.map_limits[1,1]),(4,4))
        
        Q = q.T @ q
        R = r.T @ r
        
        # Ensure R is not singular
        R = R + 1e-6*np.eye(4)

        # Set time range
        time_range = (self.current_time, self.current_time + self.t_step)
     
        # Simulate dynamics
        sol = dd.simulate(self.state, time_range, Q, R, self.A, self.B, self.u_max)
        
        # Update state
        self.current_time = sol.t[-1]
        self.state = sol.y[:,-1]
        
        self.attitude = np.hstack((np.atleast_2d(self.attitude), sol.y[0:3,:] + self.desired_angle))
        self.spin = np.hstack((np.atleast_2d(self.spin), sol.y[3:6,:]))
        self.t = np.hstack((self.t, sol.t))
       
        normalised_state = self.normalise_state(self.state)

        # Check if converged
        converged = False
        terminated = False
        truncated = False

        if sol.t_events[0].size != 0:
            converged = True
            terminated = True
        if self.current_time >= self.max_duration:
            truncated = True

        reward = 0
        settling_cost = 0
        overshoot_cost = 0

        #$$ Reward Calculation
        attitude_error = -(self.attitude[0:3,:] - self.desired_angle) 
        if terminated:
            reward = -(integrate.simps((attitude_error[0,:]**2 + attitude_error[1,:]**2 + attitude_error[2,:]**2), self.t) * 10 + self.current_time) * 10
        if truncated:
            reward = -(integrate.simps((attitude_error[0,:]**2 + attitude_error[1,:]**2 + attitude_error[2,:]**2), self.t) * 20 + self.current_time) * 10
        #$$
        
        # Return
        info = {
                'attitude': self.attitude,
                'spin': self.spin,
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

        if(options.get('deterministic', 0) == 1):
            self.deterministic_state()
        elif(self.variance_type == 'none'):
            self.deterministic_state()
        elif(self.variance_type == 'percentage'):
            self.var_state()

        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("State: ", self.state)

        # Reset state vars
        self.current_time = 0
        self.num_updates = 0
        self.initial_time = 0
        self.t = np.zeros(1)

        self.attitude = np.atleast_2d(self.initial_state[0:3] + self.desired_angle).T
        self.spin = np.atleast_2d(self.initial_state[3:6]).T
       
        self.episode_options = {}

        return self.state, {}
    

    def normalise_state (self, state):
        # Normalise the state
        normalised_state = np.zeros(state.shape)

        normalised_state[0] = state[0]/self.max_attitude
        normalised_state[1] = state[1]/self.max_attitude
        normalised_state[2] = state[2]/self.max_attitude
        normalised_state[3] = state[3]/self.max_spin
        normalised_state[4] = state[4]/self.max_spin
        normalised_state[5] = state[5]/self.max_spin
        
        if self.absolute_norm:
            normalised_state = np.abs(normalised_state)
        
        return normalised_state


    def linear_map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min # LINEAR
    
    def log_map_range(self, val, in_min, in_max, out_min, out_max):
        return out_min * (out_max/out_min)**((val - in_min)/(in_max - in_min)) # LOG
    
    def var_state(self):
        r = np.random.normal(-self.variance, self.variance, 6)
        self.state = self.initial_state*(1+r)
        
    def deterministic_state(self):
        self.state = self.initial_state
        
    def set_episode_options(self, options):
        self.episode_options = options
        
    def update_u_max(self, u_max):
        self.u_max = u_max