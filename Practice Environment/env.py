import gymnasium as gym
import numpy as np
import math
import time

from gymnasium import spaces
from gymnasium import Env

class testEnv(Env):
    
    def __init__(self, render_mode = "human"):
        
        # Define Inital State parameters
        self.initialState = np.array([
            100, # x
            100, # y
            100, # z
            0, # xdot
            0, # ydot
            0, # zdot
        ])

        # Initialize State
        self.state = self.initialState
        self.max_steps = 1000
        self.timesteps = 0

        # Set render mode
        self.render_mode = render_mode

        # Define Action/Observation Space
        action = np.array([
            1, # dx
            1, # dy
            1  # dz
        ])
        self.action_space = spaces.Box(np.negative(action), action, dtype=np.float32)

        observation = np.array([
            1000000, # x
            1000000, # y
            1000000, # z
            10000, # xdot   
            10000, # ydot
            10000  # zdot
        ])
        self.observation_space = spaces.Box(np.negative(observation), observation, dtype=np.float32)

    def step(self, action):
        self.prev_state = self.state

        # Extract State
        x, y, z, xdot, ydot, zdot = self.state
        dx, dy, dz = action

        # Update State
        xdot += dx
        ydot += dy
        zdot += dz

        x += xdot
        y += ydot
        z += zdot

        self.state = np.array([x, y, z, xdot, ydot, zdot])
        self.timesteps += 1

        # Check if Terminated
        terminated = bool(
            x == 0 and y == 0 and z == 0
        )

        truncated = bool(
            self.timesteps >= self.max_steps or
            np.linalg.norm(self.state[0:2]) > 1000
        )

        # Calculate Reward
        if not truncated or not terminated:
            if self.timesteps != 0:
                reward = -np.linalg.norm(self.state - self.prev_state)
            else:
                reward = 0
        else:
            reward = -10000

        # Render
        self.render()

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed = None, options = None):
        self.state = self.initialState
        self.timesteps = 0

        print(self.action_space)
        self.render()

        return np.array(self.state), {}
    
    def render(self):
        # Extract State
        if self.render_mode == "human":
            x, y, z, xdot, ydot, zdot = self.state
            print(f"Position: {x}, {y}, {z} | Velocity: {xdot}, {ydot}, {zdot}")