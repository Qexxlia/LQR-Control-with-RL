import math

import gymnasium as gym
import numpy as np
from scipy import integrate, linalg

import spacecraft_dynamics as scd


class SpacecraftEnv(gym.Env):

    def __init__(self, verbose, args=None):
        # Copy parameters
        self.t_step_limits = args.get("t_step_limits")
        self.variance = args.get("variance")
        self.variance_type = args.get("variance_type")
        self.max_duration = args.get("max_duration")
        self.map_limits = args.get("map_limits")
        self.u_max = args.get("u_max")
        self.t_step = args.get("t_step")
        self.seed = args.get("seed")
        self.absolute_norm = args.get("absolute_norm")
        self.verbose = verbose

        # Define the initial state
        self.initial_state = np.array(
            [
                0.5,  # x
                -0.5,  # y
                0,  # z
                1e-3,  # x_dot
                -1e-3,  # y_dot
                0,  # z_dot
                30,  # mass
            ],
            dtype=np.float64,
        )

        self.satellite_mass = 15

        # Calculate A matrix for the dynamics
        self.A, self.B = scd.precalcMatrices(6371 + 500, 3.986e5)

        # Calculate the maximum position and velocity for normalisation
        self.max_pos = 1
        self.max_vel = 0.005

        # Define the action space
        action_limits = np.ones(9, dtype=np.float64)

        self.action_space = gym.spaces.Box(
            low=-action_limits, high=action_limits, dtype=np.float64
        )

        boolabs = 0

        # Define the observation space
        observation_limits = np.ones(7, dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=boolabs * observation_limits, high=observation_limits, dtype=np.float64
        )

        # Define timeframe
        self.current_time = 0
        self.initial_time = 0

        # Define deltaV usage
        self.deltaV = 0

        # Tracking position
        self.pos = np.atleast_2d(self.initial_state[0:3]).T
        self.vel = np.atleast_2d(self.initial_state[3:6]).T

        # Set the random seed
        np.random.seed(self.seed)

        # Episode options
        self.episode_options = {}

    def step(self, action):
        # Define action
        q_weights = self.map_range(
            action[0:6], -1, 1, self.map_limits[0, 0:6], self.map_limits[1, 0:6]
        )
        r_weights = self.map_range(
            action[6:9], -1, 1, self.map_limits[0, 6:9], self.map_limits[1, 6:9]
        )
        # t_step = self.map_range(action[9], -1, 1, self.t_step_limits[0], self.t_step_limits[1])
        t_step = self.t_step_limits[0]

        if self.verbose == 1:
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("qWeights:", q_weights)
                print("rWeights:", r_weights)
                print("tStep:", t_step)
                print("State: ", self.state)

        # Set time range
        time_range = (self.current_time, self.current_time + t_step)

        # Simulate dynamics
        sol = scd.simulate(
            self.state,
            time_range,
            q_weights,
            r_weights,
            self.A,
            self.B,
            self.u_max,
            self.satellite_mass,
        )

        # if sol is None:
        #     return self.normalise_state(self.state), -1e9, False, True, {}

        # Update state
        self.current_time = sol.t[-1]
        deltaV = sol.y[6, -1] - sol.y[6, 0]
        self.deltaV = self.deltaV + deltaV
        self.state = sol.y[:, -1]

        self.pos = np.hstack((np.atleast_2d(self.pos), sol.y[0:3, :]))
        self.vel = np.hstack((np.atleast_2d(self.vel), sol.y[3:6, :]))
        self.t = np.hstack((self.t, sol.t))

        normalised_state = self.normalise_state(self.state)

        # Check if converged
        terminated = False
        truncated = False
        converged = False

        if sol.t_events[0].size != 0:
            terminated = True
        elif sol.t_events[1].size != 0:
            truncated = True
        if self.current_time >= self.max_duration:
            truncated = True

        truncated_punishment = 0
        velocity_punishment = 0
        distance_punishment = 0
        time_punishment = 0
        deltaV_punishment = 0
        reward = 0

        # $$ Reward
        reward = -integrate.simpson(
            (sol.y[0, :] ** 2 + sol.y[1, :] ** 2 + sol.y[2, :] ** 2), x=sol.t
        )
        reward -= 1
        # $$

        reward += deltaV_punishment + truncated_punishment

        # Return
        info = {
            "pos": self.pos,
            "vel": self.vel,
            "time_punishment": time_punishment,
            "deltaV_punishment": deltaV_punishment,
            "distance_punishment": distance_punishment,
            "velocity_punishment": velocity_punishment,
            "truncated_punishment": truncated_punishment,
            "t": self.t,
            "mass": self.state[6],
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
        elif self.variance_type == "range":
            self.range_state()
        elif self.variance_type == "percentage":
            self.var_state()

        if self.verbose == 1:
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("State: ", self.state)

        # Reset state vars
        self.deltaV = 0
        self.current_time = 0
        self.num_updates = 0
        self.initial_time = 0
        self.t = np.zeros(1)
        self.pos = np.atleast_2d(self.initial_state[0:3]).T
        self.vel = np.atleast_2d(self.initial_state[3:6]).T

        self.episode_options = {}

        return self.state, {}

    def normalise_state(self, state):
        # Normalise the state
        normalised_state = np.zeros(state.shape)

        normalised_state[0] = state[0] / self.max_pos
        normalised_state[1] = state[1] / self.max_pos
        normalised_state[2] = state[2] / self.max_pos
        normalised_state[3] = state[3] / self.max_vel
        normalised_state[4] = state[4] / self.max_vel
        normalised_state[5] = state[5] / self.max_vel
        normalised_state[6] = state[6] / self.initial_state[6]

        normalised_state = np.abs(normalised_state)

        return normalised_state

    def map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        # if(self.episode_options.get('log', 0) == 0):
        # return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
        # else:
        return out_min * (out_max / out_min) ** (
            (val - in_min) / (in_max - in_min)
        )  # LOG

    def var_state(self):
        r = np.append(np.random.normal(-self.variance, self.variance, 6), 0)
        self.state = self.initial_state * (1 + r)

    def deterministic_state(self):
        self.state = self.initial_state

    def range_state(self):
        # Update state based on a radius from initial conditions and random angles from initial conditions
        r = math.sqrt(
            self.initial_state[0] ** 2
            + self.initial_state[1] ** 2
            + self.initial_state[2] ** 2
        )
        theta_initial = math.atan2(self.initial_state[1], self.initial_state[0])
        phi_initial = math.acos(self.initial_state[2] / r)

        theta = np.random.uniform(
            theta_initial - self.variance, theta_initial + self.variance
        )
        phi = np.random.uniform(
            phi_initial - self.variance, phi_initial + self.variance
        )

        v = math.sqrt(
            self.initial_state[3] ** 2
            + self.initial_state[4] ** 2
            + self.initial_state[5] ** 2
        )
        gamma_initial = math.atan2(self.initial_state[4], self.initial_state[3])
        alpha_initial = math.acos(self.initial_state[5] / v)

        gamma = np.random.uniform(
            gamma_initial - self.variance, gamma_initial + self.variance
        )
        alpha = np.random.uniform(
            alpha_initial - self.variance, alpha_initial + self.variance
        )

        self.state = np.array(
            [0, 0, 0, 0, 0, 0, self.initial_state[6]], dtype=np.float64
        )
        self.state[0] = r * math.sin(theta) * math.cos(phi)
        self.state[1] = r * math.sin(theta) * math.sin(phi)
        self.state[2] = r * math.cos(theta)

        self.state[3] = v * math.sin(gamma) * math.cos(alpha)
        self.state[4] = v * math.sin(gamma) * math.sin(alpha)
        self.state[5] = v * math.cos(gamma)

    def set_episode_options(self, options):
        self.episode_options = options

    def update_u_max(self, u_max):
        self.u_max = u_max
