from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import os
import matplotlib.pyplot as plt
import SpacecraftDynamics as scd
import numpy as np

class PlotCallback(BaseCallback):
    def __init__(self, verbose=0, save_path = "./Plots/"):
        super().__init__(verbose)

        self.verbose = verbose
        self.save_path = save_path
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_rollout_end(self):
        vec_env = self.model.get_env()
        obs = vec_env.reset()
        
        done = False
            
        while not done:
            [action, _states] = self.model.predict(obs, deterministic=True)
            [obs, reward, done, info] = vec_env.step(action)
            
        self.logger.record("Q1", action[0])
        self.logger.record("Q2", action[1])
        self.logger.record("Q3", action[2])
        self.logger.record("Q4", action[3])
        self.logger.record("Q5", action[4])
        self.logger.record("Q6", action[5])

        pos = info[0].get('pos')
        vel = info[0].get('vel')
        t = info[0].get('t')

        fig = scd.plot1(pos, vel, t)
        
        name = "figure_" + str(self.num_timesteps)
        
        self.logger.record(name, Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))

        return True
    
    def _on_step(self):
        return True
