from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import os
import matplotlib.pyplot as plt
import SpacecraftDynamics as scd
import numpy as np
import pandas as pd

class PlotCallback(BaseCallback):
    def __init__(self, verbose=0, csv_save_path = "./Data/"):
        super().__init__(verbose)

        self.verbose = verbose
        self.csv_save_path = csv_save_path

        os.makedirs(self.csv_save_path, exist_ok=True)
        
    def _on_rollout_end(self):
        vec_env = self.model.get_env()
        obs = vec_env.reset()
        
        [action, _state] = self.model.predict(obs, deterministic=True)

        self.logger.record("action/q1", vec_env.env_method("map_range", action[0,0], -1, 1, 1e-16, 1e8)[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q2", vec_env.env_method("map_range", action[0,1], -1, 1, 1e-16, 1e8)[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q3", vec_env.env_method("map_range", action[0,2], -1, 1, 1e-16, 1e8)[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q4", vec_env.env_method("map_range", action[0,3], -1, 1, 1e-16, 1e8)[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q5", vec_env.env_method("map_range", action[0,4], -1, 1, 1e-16, 1e8)[0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q6", vec_env.env_method("map_range", action[0,5], -1, 1, 1e-16, 1e8)[0], exclude=("stdout", "log", "json", "csv"))
       
        done = False
        while not done:
            [obs, reward, done, info] = vec_env.step(action)
            [action, _state] = self.model.predict(obs, deterministic=True)
            
        pos = info[-1].get('pos')
        vel = info[-1].get('vel')
        t = info[-1].get('t')

        # Plot the state and output to tensorboard
        fig = scd.plot1(pos, vel, t)
        self.logger.record("rollout/ep_state_figure", Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))
        
        # Create a DataFrame and export to CSV
        data = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'vx': vel[0], 'vy': vel[1], 'vz': vel[2], 't': t}
        df = pd.DataFrame(data)
        name = self.csv_save_path + "state" + str(self.num_timesteps) + ".csv"
        df.to_csv(name, index=False)
        
        # Save deltaV and time to Tensorboard
        time = t[-1]
        deltaV = info[-1].get('dVT')

        self.logger.record("rollout/ep_delta_v", deltaV, exclude=("stdout", "log", "json", "csv"))
        self.logger.record("rollout/ep_time_elapsed", time, exclude=("stdout", "log", "json", "csv"))
    
        return True
    
    def _on_step(self):
        return True
