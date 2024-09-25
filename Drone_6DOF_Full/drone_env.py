import gymnasium as gym
import numpy as np

import drone_dynamics as dd


class DroneEnv(gym.Env):

    def __init__(self, verbose, args=None):
        # Copy parameters
        self.t_step = args.get("t_step")
        self.max_duration = args.get("max_duration")
        self.map_limits = args.get("map_limits")
        self.u_max = args.get("u_max")
        self.seed = args.get("seed")
        self.verbose = verbose

        # Define the initial state
        self.initial_state = np.array(
            [
                10,
                7,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )

        # Calculate A matrix for the dynamics
        (self.A, self.B) = dd.precalcMatrices()

        # Calculate the maximum position and velocity for normalisation
        self.max_attitude = 20
        self.max_spin = 400
        self.max_position = 10
        self.max_velocity = 40

        # Define the action space
        self.action_space = gym.spaces.Box(
            low=-1 * np.ones(88), high=np.ones(88), dtype=np.float32
        )

        # Define the observation space
        self.observation_space = gym.spaces.Box(
            low=np.zeros(12), high=np.ones(12), dtype=np.float32
        )

        # Define timeframe
        self.current_time = 0
        self.initial_time = 0

        # Tracking position
        self.position = np.atleast_2d(self.initial_state[0:3]).T
        self.attitude = np.atleast_2d(self.initial_state[3:6]).T
        self.velocity = np.atleast_2d(self.initial_state[6:9]).T
        self.spin = np.atleast_2d(self.initial_state[9:12]).T
        self.u = np.atleast_2d(np.zeros(4, dtype=np.float32)).T

        # Action masks
        self.q_off_diagonal_mask = np.ones((12, 12), dtype=bool)
        self.q_off_diagonal_mask[np.triu_indices(12)] = False
        self.q_diagonal_mask = np.eye(12, dtype=bool)

        self.r_off_diagonal_mask = np.ones((4, 4), dtype=bool)
        self.r_off_diagonal_mask[np.triu_indices(4)] = False
        self.r_diagonal_mask = np.eye(4, dtype=bool)

        # Set the random seed
        np.random.seed(self.seed)

        # Episode options
        self.episode_options = {}

    def step(self, action):
        # Define action
        Q, R = self.map_action(action)
        if Q is None:
            normalised_state = self.normalise_state(self.state)
            return normalised_state, -1e9, False, True, {}

        # Set time range
        time_range = (self.current_time, self.current_time + self.t_step)

        # Simulate dynamics
        sol, u = dd.simulate(self.state, time_range, Q, R, self.A, self.B)

        if sol is None:
            normalised_state = self.normalise_state(self.state)
            return normalised_state, -1e9, False, True, {}

        if len(sol.t_events[1]) != 0:
            normalised_state = self.normalise_state(self.state)
            reward = -1e3 * (1 + np.sum(abs(sol.y[:, -1])))
            return normalised_state, reward, False, True, {}

        # Update state
        self.current_time = sol.t[-1]
        self.state = sol.y[:, -1]

        self.attitude = np.hstack((np.atleast_2d(self.attitude), sol.y[3:6, :]))
        self.spin = np.hstack((np.atleast_2d(self.spin), sol.y[9:12, :]))
        self.position = np.hstack((np.atleast_2d(self.position), sol.y[0:3, :]))
        self.velocity = np.hstack((np.atleast_2d(self.velocity), sol.y[6:9:]))
        self.t = np.hstack((self.t, sol.t))
        self.u = np.hstack((self.u, u))

        normalised_state = self.normalise_state(self.state)

        # Check if converged
        terminated = False
        truncated = False

        if sol.t_events[0].size != 0:
            terminated = True
        if self.current_time >= self.max_duration:
            truncated = True

        # $$ Reward Calculation
        reward = 0
        if terminated or truncated:
            reward = -self.current_time
        if truncated:
            reward -= np.sum(abs(sol.y[:, -1]))

        # $$

        # Return
        info = {
            "attitude": self.attitude,
            "spin": self.spin,
            "position": self.position,
            "velocity": self.velocity,
            "u": self.u,
            "t": self.t,
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

        self.deterministic_state()

        if self.verbose == 1:
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print("State: ", self.state)

        # Reset state vars
        self.current_time = 0
        self.num_updates = 0
        self.initial_time = 0
        self.t = np.zeros(1)

        self.episode_options = {}

        self.position = np.atleast_2d(self.initial_state[0:3]).T
        self.attitude = np.atleast_2d(self.initial_state[3:6]).T
        self.velocity = np.atleast_2d(self.initial_state[6:9]).T
        self.spin = np.atleast_2d(self.initial_state[9:12]).T
        self.u = np.atleast_2d(np.zeros(4, dtype=np.float32)).T

        return self.state, {}

    def normalise_state(self, state):
        # Normalise the state
        normalised_state = np.zeros(state.shape)

        normalised_state[0] = state[0] / self.max_position
        normalised_state[1] = state[1] / self.max_position
        normalised_state[2] = state[2] / self.max_position
        normalised_state[3] = state[3] / self.max_attitude
        normalised_state[4] = state[4] / self.max_attitude
        normalised_state[5] = state[5] / self.max_attitude
        normalised_state[6] = state[6] / self.max_velocity
        normalised_state[7] = state[7] / self.max_velocity
        normalised_state[8] = state[8] / self.max_velocity
        normalised_state[9] = state[9] / self.max_spin
        normalised_state[10] = state[10] / self.max_spin
        normalised_state[11] = state[11] / self.max_spin

        normalised_state = np.abs(normalised_state)

        return normalised_state

    def linear_map_range(self, val, in_min, in_max, out_min, out_max):
        # Map a value from one range to another
        return (val - in_min) / (in_max - in_min) * (
            out_max - out_min
        ) + out_min  # LINEAR

    def log_map_range(self, val, in_min, in_max, out_min, out_max, neg):
        if neg:
            return np.copysign(
                out_min
                * (out_max / out_min) ** ((abs(val) - in_min) / (in_max - in_min)),
                val,
            )  # LOG
        else:
            return out_min * (out_max / out_min) ** (
                (abs(val) - in_min) / (in_max - in_min)
            )

    def var_state(self):
        r = np.random.normal(-self.variance, self.variance, 6)
        self.state = self.initial_state * (1 + r)

    def deterministic_state(self):
        self.state = self.initial_state

    def set_episode_options(self, options):
        self.episode_options = options

    def update_u_max(self, u_max):
        self.u_max = u_max

    def map_action(self, action):
        q_off_diagonal_weights = self.linear_map_range(
            action[0:66],
            -1,
            1,
            np.ones(66, dtype=np.float32) * self.map_limits[0],
            np.ones(66, dtype=np.float32) * self.map_limits[1],
        )

        q_diagonal_weights = self.linear_map_range(
            action[66:78],
            -1,
            1,
            np.ones(12, dtype=np.float32) * self.map_limits[0],
            np.ones(12, dtype=np.float32) * self.map_limits[1],
        )

        r_off_diagonal_weights = self.linear_map_range(
            action[78:84],
            -1,
            1,
            np.ones(6, dtype=np.float32) * self.map_limits[0],
            np.ones(6, dtype=np.float32) * self.map_limits[1],
        )

        r_diagonal_weights = self.linear_map_range(
            action[84:88],
            -1,
            1,
            np.ones(4, dtype=np.float32) * self.map_limits[0],
            np.ones(4, dtype=np.float32) * self.map_limits[1],
        )

        q = np.zeros((12, 12))
        q[self.q_off_diagonal_mask] = q_off_diagonal_weights / 1000
        q[self.q_diagonal_mask] = q_diagonal_weights

        r = np.zeros((4, 4))
        r[self.r_off_diagonal_mask] = r_off_diagonal_weights / 1000
        r[self.r_diagonal_mask] = r_diagonal_weights

        Q = q @ q.T
        R = r @ r.T

        min_svd = np.linalg.svd(R, compute_uv=False)[-1]
        if min_svd == 0 or min_svd < np.spacing(1.0) * np.linalg.norm(R, 1):
            return None, None

        return Q, R
