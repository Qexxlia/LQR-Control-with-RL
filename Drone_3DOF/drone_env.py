import math

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy import integrate

import drone_dynamics_verbose as dd


class DroneEnv(gym.Env):

    def __init__(self, verbose, args=None):
        # Copy parameters
        self.t_step_limits = args.get("t_step_limits")
        self.variance = args.get("variance")
        self.variance_type = args.get("variance_type")
        self.max_duration = args.get("max_duration")
        self.map_limits = args.get("map_limits")
        self.u_max = args.get("u_max")
        self.seed = args.get("seed")
        self.absolute_norm = args.get("absolute_norm")
        self.verbose = verbose

        self.desired_angle = 0.0872664625997165

        # Define the initial state
        self.initial_state = np.array(
            [
                -self.desired_angle,  # r
                -self.desired_angle,  # p
                -self.desired_angle,  # y
                0,  # dr
                0,  # dp
                0,  # dy
            ],
            dtype=np.float32,
        )

        # Calculate A matrix for the dynamics
        (self.A, self.B) = dd.precalcMatrices(
            0.0036, 0.1188, 0.1969, 0.0552, 0.0552, 0.1104
        )

        # Calculate the maximum position and velocity for normalisation
        self.max_attitude = 2 * self.desired_angle
        self.max_spin = 6 * self.desired_angle

        # Define the action space
        action_limits = np.ones(8, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-action_limits, high=action_limits, dtype=np.float32
        )

        # Change the observation space if absolute normalisation is used
        # if self.absolute_norm:
        boolabs = 0
        # else:
        # boolabs = -1

        # Define the observation space
        observation_limits = np.ones(6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=boolabs * observation_limits, high=observation_limits, dtype=np.float32
        )

        # Define timeframe
        self.current_time = 0
        self.initial_time = 0

        self.t_step = 0

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
        q_weights = self.map_range(
            action[0:6], -1, 1, self.map_limits[0, 0:6], self.map_limits[1, 0:6]
        )
        r_weight = self.map_range(
            action[6], -1, 1, self.map_limits[0, 6], self.map_limits[1, 6]
        )
        self.t_step = self.map_range(
            action[7], -1, 1, self.t_step_limits[0], self.t_step_limits[1]
        )
        r_weights = np.array([r_weight, r_weight, r_weight, r_weight])

        if self.verbose == 1:
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("qWeights:", q_weights)
                print("rWeights:", r_weights)
                print("tStep:", self.t_step)
                print("State: ", self.state)

        # Set time range
        time_range = (self.current_time, self.current_time + self.t_step)

        # Simulate dynamics
        if self.verbose == 1:
            print("Simulating...")
        [sol, u] = dd.simulate(
            self.state, time_range, q_weights, r_weights, self.A, self.B, self.u_max
        )
        if self.verbose == 1:
            print("Simulated")

        # Update state
        self.current_time = sol.t[-1]
        self.state = sol.y[:, -1]

        self.u = np.hstack((self.u, u))

        self.attitude = np.hstack(
            (np.atleast_2d(self.attitude), sol.y[0:3, :] + self.desired_angle)
        )
        self.spin = np.hstack((np.atleast_2d(self.spin), sol.y[3:6, :]))
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

        # $$ Reward Calculation
        attitude_error = -(self.attitude[0:3, :] - self.desired_angle)
        if terminated:
            reward = (
                -(
                    integrate.simpson(
                        (
                            attitude_error[0, :] ** 2
                            + attitude_error[1, :] ** 2
                            + attitude_error[2, :] ** 2
                        ),
                        x=self.t,
                    )
                    * 10
                    + self.current_time
                )
                * 10
            )
        if truncated:
            reward = (
                -(
                    integrate.simps(
                        (
                            attitude_error[0, :] ** 2
                            + attitude_error[1, :] ** 2
                            + attitude_error[2, :] ** 2
                        ),
                        self.t,
                    )
                    * 20
                    + self.current_time
                )
                * 10
            )
        # $$

        # Return
        info = {
            "attitude": self.attitude,
            "spin": self.spin,
            "t": self.t,
            "settling_cost": settling_cost,
            "overshoot_cost": overshoot_cost,
            "u": self.u,
        }

        return normalised_state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options={}):
        # Reset state
        options = self.episode_options

        # Allow single verbose run
        if options.get("verbose", 0) == 1:
            self.verbose = 1
        else:
            self.verbose = 0

        if options.get("deterministic", 0) == 1:
            self.deterministic_state()
        elif self.variance_type == "none":
            self.deterministic_state()
        elif self.variance_type == "percentage":
            self.var_state()

        if self.verbose == 1:
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

    def normalise_state(self, state):
        # Normalise the state
        normalised_state = np.zeros(state.shape)

        normalised_state[0] = state[0] / self.max_attitude
        normalised_state[1] = state[1] / self.max_attitude
        normalised_state[2] = state[2] / self.max_attitude
        normalised_state[3] = state[3] / self.max_spin
        normalised_state[4] = state[4] / self.max_spin
        normalised_state[5] = state[5] / self.max_spin

        # if self.absolute_norm:
        normalised_state = np.abs(normalised_state)

        return normalised_state

    def map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        # return (val - in_min)/(in_max - in_min)*(out_max - out_min) + out_min # LINEAR
        return out_min * (out_max / out_min) ** (
            (val - in_min) / (in_max - in_min)
        )  # LOG

    def var_state(self):
        r = np.random.normal(-self.variance, self.variance, 6)
        self.state = self.initial_state * (1 + r)

    def deterministic_state(self):
        self.state = self.initial_state

    def set_episode_options(self, options):
        self.episode_options = options

    def update_u_max(self, u_max):
        self.u_max = u_max

