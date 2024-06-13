from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.logger import HParam
from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import SpacecraftDynamics as scd
import numpy as np
import pandas as pd

class StateCallback(BaseCallback):
    def __init__(self, csv_save_path, map_limits, max_duration):
        super().__init__(0)

        self.csv_save_path = csv_save_path
        self.map_limits = map_limits
        self.max_duration = max_duration

        os.makedirs(self.csv_save_path, exist_ok=True)
        
    def _on_rollout_end(self):
        vec_env = self.model.get_env()
        obs = vec_env.reset()
            
        [action, _state] = self.model.predict(obs, deterministic=True)

        # Save initial actions to tensorboard
        qWeights = vec_env.env_method("map_range", action[0,0:6], -1, 1, self.map_limits[0,0:6], self.map_limits[1,0:6])[0]
        rWeights = vec_env.env_method("map_range", action[0,6:9], -1, 1, self.map_limits[0,6:9], self.map_limits[1,6:9])[0]

        self.logger.record("action/q0", qWeights[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q1", qWeights[1], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q2", qWeights[2], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q3", qWeights[3], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q4", qWeights[4], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q5", qWeights[5], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/r0", rWeights[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/r1", rWeights[1], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/r2", rWeights[2], exclude=("stdout", "log", "json", "csv"))

        done = False
        while not done:
            [obs, reward, done, info] = vec_env.step(action)
            [action, _state] = self.model.predict(obs, deterministic=True)
            if(info[-1].get('t')[-1] > self.max_duration):
                done = True
            
        pos = info[-1].get('pos')
        vel = info[-1].get('vel')
        t = info[-1].get('t')
        reward = reward[-1]

        # Plot the state and output to tensorboard
        fig = scd.plot1(pos, vel, t)
        self.logger.record("plots/ep_state_figure", Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))
        
        # Create a DataFrame and export to CSV
        data = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'vx': vel[0], 'vy': vel[1], 'vz': vel[2], 't': t}
        df = pd.DataFrame(data)
        name = self.csv_save_path + "state" + str(self.num_timesteps) + ".csv"
        df.to_csv(name, index=False)
        
        # Save deltaV and time to Tensorboard
        time = t[-1]
        deltaV = info[-1].get('dVT')
        velocity_error = np.linalg.norm(vel, -1)
        position_error = np.linalg.norm(pos, -1)

        self.logger.record("reward/ep_delta_v", deltaV, exclude=("stdout", "log", "json", "csv"))
        self.logger.record("reward/ep_time_elapsed", time, exclude=("stdout", "log", "json", "csv"))
        self.logger.record("reward/ep_position_error", position_error, exclude=("stdout", "log", "json", "csv"))
        self.logger.record("reward/ep_velocity_error", velocity_error, exclude=("stdout", "log", "json", "csv"))
        self.logger.record("reward/ep_reward", reward, exclude=("stdout", "log", "json", "csv"))
    
        return True
    
    def _on_step(self):
        return True


class HParamCallback(BaseCallback):
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        hparam_dict = {
           "algorithm": self.model.__class__.__name__,
           "learning rate": self.model.learning_rate,
           "n steps": self.model.n_steps,
           "n epochs": self.model.n_epochs,
           "batch size": self.model.batch_size,
           "gamma": self.model.gamma,
           "entropy coefficient": self.model.ent_coef,
           "value function coefficient": self.model.vf_coef,
           "max gradient norm": self.model.max_grad_norm,
           "initial std (log)": self.model.policy_kwargs.get('log_std_init')
        }

        metric_dict = {
            "train/clip_range": 0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
        