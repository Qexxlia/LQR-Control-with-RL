from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import os
import matplotlib.pyplot as plt
import SpacecraftDynamics as scd
import numpy as np

class PlotCallback(BaseCallback):
    def __init__(self, verbose=0, save_path = "./Figures/"):
        super().__init__(verbose)

        self.verbose = verbose
        self.save_path = save_path
        
    def _on_rollout_end(self):
        vec_env = self.model.get_env()
        obs = vec_env.reset()
        
        done = False
            
        [action, _states] = self.model.predict(obs, deterministic=True)
        self.logger.record("action/q1", action[0,0], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q2", action[0,1], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q3", action[0,2], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q4", action[0,3], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q5", action[0,4], exclude=("stdout", "log", "json", "csv"))
        self.logger.record("action/q6", action[0,5], exclude=("stdout", "log", "json", "csv"))

        while not done:
            [obs, reward, done, info] = vec_env.step(action)
            [action, _states] = self.model.predict(obs, deterministic=True)
            

        pos = info[0].get('pos')
        vel = info[0].get('vel')
        t = info[0].get('t')

        fig = scd.plot1(pos, vel, t)
        
        self.logger.record("figures/state", Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))

        return True
    
    def _on_step(self):
        return True
