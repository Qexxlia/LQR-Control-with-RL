import gymnasium as gym
import numpy as np
import pandas as pd
import math

import SpacecraftDynamics as scd

class SpacecraftEnv(gym.Env):
    
    def __init__(self, verbose, args = None):
        # Copy parameters
        self.t_step_limits = args.get('t_step_limits')
        self.variance = args.get('variance')
        self.variance_type = args.get('variance_type')
        self.max_duration = args.get('max_duration')
        self.map_limits = args.get('map_limits')
        self.u_max = args.get('u_max')
        self.verbose = verbose

        # Define the inital state
        self.initial_state = np.array([
            0.5,    # x
            -0.5,    # y
            0,    # z
            1e-3,   # x_dot
            -1e-3,   # y_dot
            0,   # z_dot
            30,   # mass 
        ], dtype=np.float32)

        self.A = scd.calcAMatrix(6371 + 500, 3.986e5)
        
        self.max_pos = max(abs(self.initial_state[0:3])) 
        self.max_vel = self.max_pos/10

        # Define the action space
        action_limits = np.ones(10, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low = -action_limits,
            high = action_limits,
            dtype = np.float32
        )
        
        # Define the observation space
        observation_limits = np.ones(7, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low = 0*observation_limits,
            high = observation_limits,
            dtype = np.float32
        )

        # Define timeframe
        self.current_time = 0
        self.initial_time = 0
        
        # Define deltaV usage
        self.deltaV = 0
        
        # Tracking Time / Weight Updates
        self.num_updates = 0
        
        # Tracking position
        self.pos = np.atleast_2d(self.initial_state[0:3]).T
        self.vel = np.atleast_2d(self.initial_state[3:6]).T
        
        np.random.seed(0)
        
        self.reset_options = {}
        
    def step(self, action):
        # Define action
        q_weights = self.map_range(action[0:6], -1, 1, self.map_limits[0,0:6], self.map_limits[1,0:6])
        r_weights = self.map_range(action[6:9], -1, 1, self.map_limits[0,6:9], self.map_limits[1,6:9])
        t_step = self.map_range(action[9], -1, 1, self.t_step_limits[0], self.t_step_limits[1])

        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("Action:", action)
                print("Q:", q_weights)
                print("R:", r_weights)
        
        timeRange = (self.current_time, self.current_time + t_step)
     
        # Simulate dynamics
        sol = scd.simulate(self.state, timeRange, q_weights, r_weights, self.A, self.u_max)
        
        self.num_updates += 1
        self.current_time = sol.t[-1]
        
        self.deltaV -= sol.y[6, -1] - sol.y[6, 0]

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
        if self.current_time >= self.max_duration:
            truncated = True
        
        #$$ Reward Calculation
        reward = 0
        time_punishment = self.current_time - self.initial_time
        deltaV_punishment = self.deltaV * 200
        distance_punishment = np.linalg.norm(sol.y[0:3, -1])
        velocity_punishment = np.linalg.norm(sol.y[3:6, -1])
        truncated_punishment = 0
        converged_reward = 0

        if no_deltaV:
            deltaV_punishment = 5000
        if truncated:
            truncated_punishment = 5000
        if converged:
            converged_reward = 5000
        
        reward = - time_punishment - deltaV_punishment - distance_punishment - velocity_punishment - truncated_punishment + converged_reward
        #$$
            
        # Update state
        self.state = sol.y[:,-1]
        
        self.pos = np.hstack((np.atleast_2d(self.pos), sol.y[0:3, :]))
        self.vel = np.hstack((np.atleast_2d(self.vel), sol.y[3:6, :]))
        self.t = np.hstack((self.t, sol.t))
       
        normalised_state = self.normalise_state(self.state)

        # Return
        return normalised_state, reward, terminated, truncated, {'pos': self.pos, 'vel': self.vel, 't': self.t, 'deltaV': self.deltaV}
    
    def reset(self, *, seed = None, options = {}):
        # Reset state
        options = self.reset_options

        if(options.get('deterministic', 0) == 1):
            self.deterministic_state()
        elif(self.variance_type == 'range'):
            self.range_state()
        elif(self.variance_type == 'percentage'):
            self.var_state()

        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("State: ", self.state)

        self.deltaV = 0

        self.current_time = 0
        self.num_updates = 0
        self.initial_time = 0
        
        self.t = np.zeros(1)
        self.pos = np.atleast_2d(self.initial_state[0:3]).T
        self.vel = np.atleast_2d(self.initial_state[3:6]).T
        
        self.reset_options = {}
        
        return self.state, {'pos': self.pos, 'vel': self.vel, 't': self.t, 'deltaV': self.deltaV}
    

    def normalise_state (self, state):
        # Normalise the state
        normalised_state = np.zeros(state.shape)

        normalised_state[0] = state[0]/self.max_pos
        normalised_state[1] = state[1]/self.max_pos
        normalised_state[2] = state[2]/self.max_pos
        normalised_state[3] = state[3]/self.max_vel
        normalised_state[4] = state[4]/self.max_vel
        normalised_state[5] = state[5]/self.max_vel
        normalised_state[6] = state[6]/self.initial_state[6]
        
        return abs(normalised_state)


    def map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min
    
    def var_state(self):
        r = np.append(np.random.normal(-self.variance, self.variance, 6), 0)
        self.state = self.initial_state*(1+r)
        
    def deterministic_state(self):
        self.state = self.initial_state
    
    def range_state(self):
        r = math.sqrt(self.initial_state[0]**2 + self.initial_state[1]**2 + self.initial_state[2]**2)
        theta = np.random.uniform(0, self.variance)
        phi = np.random.uniform(0, self.variance)

        v = math.sqrt(self.initial_state[3]**2 + self.initial_state[4]**2 + self.initial_state[5]**2)
        gamma = np.random.uniform(0, self.variance)
        alpha = np.random.uniform(0, self.variance)

        self.state = np.array([0, 0, 0, 0, 0, 0, self.initial_state[6]], dtype=np.float32)
        self.state[0] = r * math.sin(theta) * math.cos(phi)
        self.state[1] = r * math.sin(theta) * math.sin(phi)
        self.state[2] = r * math.cos(theta)

        self.state[3] = v * math.sin(gamma) * math.cos(alpha)
        self.state[4] = v * math.sin(gamma) * math.sin(alpha)
        self.state[5] = v * math.cos(gamma)
        
    def set_reset_options(self, options):
        self.reset_options = options