import gymnasium as gym
import numpy as np
import pandas as pd
import math

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
        self.simulation_type = args.get('simulation_type')
        self.t_step = args.get('t_step')
        self.seed = args.get('seed')
        self.absolute_norm = args.get('absolute_norm')
        self.verbose = verbose

        # Define the initial state
        self.initial_state = np.array([
            1, # x
            1, # y
            0, # z
            0, # roll
            0, # pitch
            0, # yaw
            0, # x_dot
            0, # y_dot
            0, # z_dot
            0, # roll_dot
            0, # pitch_dot
            0, # yaw_dot
        ], dtype=np.float32)
        
        # Calculate A matrix for the dynamics
        (self.A, self.B) = dd.precalcMatrices(1, 0.11, 0.11, 0.04) # From F. Ahmed et al
        
        # Calculate the maximum position and velocity for normalisation
        self.max_position = 1
        self.max_attitude = np.pi/8
        self.max_velocity = 0.5
        self.max_spin = np.pi/16

        # Define the action space
        match self.simulation_type:
            case 'q':
                action_limits = np.ones(12, dtype=np.float32)
            case 'qt':
                action_limits = np.ones(13, dtype=np.float32)
            case 'r':
                action_limits = np.ones(4, dtype=np.float32)
            case 'rt':
                action_limits = np.ones(5, dtype=np.float32)
            case 'qr':
                action_limits = np.ones(16, dtype=np.float32)
            case 'qrt':
                action_limits = np.ones(17, dtype=np.float32)
            case _:
                raise ValueError("Invalid simulation type")

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
        observation_limits = np.ones(12, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low = boolabs*observation_limits,
            high = observation_limits,
            dtype = np.float32
        )

        # Define timeframe
        self.current_time = 0
        self.initial_time = 0
        
        # Define deltaV usage
        self.deltaV = 0
        
        # Tracking position
        self.position = np.atleast_2d(self.initial_state[0:3]).T
        self.attitude = np.atleast_2d(self.initial_state[3:6]).T
        self.velocity = np.atleast_2d(self.initial_state[6:9]).T
        self.spin = np.atleast_2d(self.initial_state[9:12]).T
        
        # Set the random seed
        np.random.seed(self.seed)
        
        # Episode options
        self.episode_options = {}

    def step(self, action):
        # Define action
        match self.simulation_type:
            case('q'):
                q_weights = self.map_range(action[0:12], -1, 1, self.map_limits[0,0:12], self.map_limits[1,0:12])
                r_weights = np.ones(3, dtype=np.float32)
                t_step = self.t_step
            case('qt'):
                q_weights = self.map_range(action[0:12], -1, 1, self.map_limits[0,0:12], self.map_limits[1,0:12])
                r_weights = np.ones(3, dtype=np.float32)
                t_step = self.map_range(action[12], -1, 1, self.t_step_limits[0], self.t_step_limits[1])
            case('r'):
                q_weights = np.ones(6, dtype=np.float32)
                r_weights = self.map_range(action[0:4], -1, 1, self.map_limits[0,12:16], self.map_limits[1,12:16])
                t_step = self.map_range(action[4], -1, 1, self.t_step_limits[0], self.t_step_limits[1])
            case('qr'):
                q_weights = self.map_range(action[0:12], -1, 1, self.map_limits[0,0:12], self.map_limits[1,0:12])
                r_weights = self.map_range(action[12:16], -1, 1, self.map_limits[0,12:16], self.map_limits[1,12:16])
                t_step = self.t_step
            case('qrt'):
                q_weights = self.map_range(action[0:12], -1, 1, self.map_limits[0,0:12], self.map_limits[1,0:12])
                r_weights = self.map_range(action[12:16], -1, 1, self.map_limits[0,12:16], self.map_limits[1,12:16])
                t_step = self.map_range(action[16], -1, 1, self.t_step_limits[0], self.t_step_limits[1])

        if(self.verbose == 1):
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("qWeights:", q_weights)
                print("rWeights:", r_weights)
                print("tStep:", t_step)
                print("State: ", self.state)
        
        # Set time range
        time_range = (self.current_time, self.current_time + t_step)
     
        # Simulate dynamics
        if self.verbose == 1: print("Simulating...")
        sol = dd.simulate(self.state, time_range, q_weights, r_weights, self.A, self.B, self.u_max)
        if self.verbose == 1: print("Simulated")
        
        # Update state
        self.current_time = sol.t[-1]
        self.state = sol.y[:,-1]
        
        self.position = np.hstack((np.atleast_2d(self.position), sol.y[0:3,:]))
        self.attitude = np.hstack((np.atleast_2d(self.attitude), sol.y[3:6,:]))
        self.velocity = np.hstack((np.atleast_2d(self.velocity), sol.y[6:9,:]))
        self.spin = np.hstack((np.atleast_2d(self.spin), sol.y[9:12,:]))
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

        truncated_punishment = 0
        velocity_punishment = 0
        distance_punishment = 0
        time_punishment = 0
        deltaV_punishment = 0
        reward = 0

        #$$ Reward Calculation
        
        if terminated:
            reward = -np.linalg.norm(self.position)
        if truncated:
            reward = -1000
        
        #$$
        
        # Return
        info = {'position': self.position, 
                'attitude': self.attitude,
                'velocity': self.velocity, 
                'spin': self.spin,
                'time_punishment': time_punishment, 
                'distance_punishment': distance_punishment, 
                'velocity_punishment': velocity_punishment, 
                'truncated_punishment': truncated_punishment, 
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

        self.position = np.atleast_2d(self.initial_state[0:3]).T
        self.attitude = np.atleast_2d(self.initial_state[3:6]).T
        self.velocity = np.atleast_2d(self.initial_state[6:9]).T
        self.spin = np.atleast_2d(self.initial_state[9:12]).T
       
        self.episode_options = {}

        return self.state, {}
    

    def normalise_state (self, state):
        # Normalise the state
        normalised_state = np.zeros(state.shape)

        normalised_state[0] = state[0]/self.max_position
        normalised_state[1] = state[1]/self.max_position
        normalised_state[2] = state[2]/self.max_position
        normalised_state[3] = state[3]/self.max_attitude
        normalised_state[4] = state[4]/self.max_attitude
        normalised_state[5] = state[5]/self.max_attitude
        normalised_state[6] = state[6]/self.max_velocity
        normalised_state[7] = state[7]/self.max_velocity
        normalised_state[8] = state[8]/self.max_velocity
        normalised_state[9] = state[9]/self.max_spin
        normalised_state[10] = state[10]/self.max_spin
        normalised_state[11] = state[11]/self.max_spin
        
        if self.absolute_norm:
            normalised_state = np.abs(normalised_state)
        
        return normalised_state


    def map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min
    
    def var_state(self):
        r = np.random.normal(-self.variance, self.variance, 12)
        self.state = self.initial_state*(1+r)
        
    def deterministic_state(self):
        self.state = self.initial_state
        
    def set_episode_options(self, options):
        self.episode_options = options
        
    def update_u_max(self, u_max):
        self.u_max = u_max