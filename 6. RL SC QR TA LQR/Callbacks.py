from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import TensorBoardOutputFormat
from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import SpacecraftDynamics as scd
import numpy as np

class StateCallback(BaseCallback):
    def __init__(self, args):
        super().__init__(0)

        self.map_limits = args.get('map_limits')
        self.max_duration = args.get('max_duration')
        self.t_step_limits = args.get('t_step_limits')
        
    def _on_rollout_end(self):
        vec_env = self.model.get_env()
        vec_env.env_method("set_reset_options", {'deterministic': 1})
        obs = vec_env.reset()
            
        [action, _state] = self.model.predict(obs, deterministic=True)
        action_record = action

        # Save initial actions to tensorboard
        # qWeights = vec_env.env_method("map_range", action[0,0:6], -1, 1, self.map_limits[0,0:6], self.map_limits[1,0:6])[0]
        # rWeights = vec_env.env_method("map_range", action[0,6:9], -1, 1, self.map_limits[0,6:9], self.map_limits[1,6:9])[0]
        # t_step = vec_env.env_method("map_range", action[0,9], -1, 1, self.t_step_limits[0], self.t_step_limits[1])[0]

        # self.logger.record("action/q0", qWeights[0], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/q1", qWeights[1], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/q2", qWeights[2], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/q3", qWeights[3], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/q4", qWeights[4], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/q5", qWeights[5], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/r0", rWeights[0], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/r1", rWeights[1], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/r2", rWeights[2], exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("action/t_step", t_step, exclude=("stdout", "log", "json", "csv"))

        done = False
        while not done:
            [obs, reward, done, info] = vec_env.step(action)
            action_record = np.vstack((action_record, action))
            [action, _state] = self.model.predict(obs, deterministic=True)
            if(info[-1].get('t')[-1] > self.max_duration):
                done = True

        pos = info[-1].get('pos')
        vel = info[-1].get('vel')
        t = info[-1].get('t')
        reward = reward[-1]

        qWeights = vec_env.env_method("map_range", action_record[:,0:6], -1, 1, self.map_limits[0,0:6], self.map_limits[1,0:6])[0]
        rWeights = vec_env.env_method("map_range", action_record[:,6:9], -1, 1, self.map_limits[0,6:9], self.map_limits[1,6:9])[0]
        t_step = vec_env.env_method("map_range", action_record[:,9], -1, 1, self.t_step_limits[0], self.t_step_limits[1])[0]

        # Plot Actions
        
        t_e = np.vstack((t,t,t)).transpose()
        
        first = True
        for step in t_step:
            if first:
                t_a = np.array([step])
                first = False
            else:
                t_a = np.append(t_a, t_a[-1]+step)
        
        t_a1 = np.vstack((t_a, t_a, t_a)).transpose()
        t_a2 = t_a
        t_a = np.vstack((t_a, t_a, t_a, t_a, t_a, t_a)).transpose()

        tsL = 12
        tsS = 7

        fig = plt.figure(1, figsize=(10,2)) 
        ax = fig.add_subplot(1,1,1)
        ax.plot(t_a, qWeights, label = ['Q0','Q1','Q2','Q3','Q4','Q5'])
        ax.set_title('Q Weights', fontsize=tsL)
        ax.set_xlabel('Time (s)', fontsize=tsS)
        ax.set_ylabel('Weight', fontsize=tsS)
        fig.legend()

        fig1 = plt.figure(2, figsize=(10,2)) 
        ax1 = fig1.add_subplot(1,1,1)
        ax1.plot(t_a1, rWeights, label = ['R0','R1','R2'])
        ax1.set_title('R Weights', fontsize=tsL)
        ax1.set_xlabel('Time (s)', fontsize=tsS)
        ax1.set_ylabel('Weight', fontsize=tsS)
        fig1.legend()

        fig2 = plt.figure(3, figsize=(10,2)) 
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(t_a2, t_step)
        ax2.set_title('Time Step', fontsize=tsL)
        ax2.set_xlabel('Time (s)', fontsize=tsS)
        ax2.set_ylabel('Time Step (s)', fontsize=tsS)
        
        # Plot Position and Velocity
        fig3 = plt.figure(4, figsize=(10,2)) 
        ax3 = fig3.add_subplot(1,1,1)
        ax3.plot(t_e, pos[0:3, :].transpose(), label=['x','y','z'])
        ax3.set_xlabel('Time (s)', fontsize=tsS)
        ax3.set_title('Position', fontsize=tsL)
        ax3.set_ylabel('Position (km)', fontsize=tsS)
        fig3.legend()

        fig4 = plt.figure(5, figsize=(10,2)) 
        ax4 = fig4.add_subplot(1,1,1)
        ax4.plot(t_e, vel[0:3, :].transpose(), label=['vx','vy','vz'])
        ax4.set_xlabel('Time (s)', fontsize=tsS)
        ax4.set_title('Velocity vs Time', fontsize=tsL)
        ax4.set_ylabel('Velocity (km/s)', fontsize=tsS)
        fig4.legend()

        # Plot to tensorboard
        self.logger.record("plots/q_weights", Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/r_weights", Figure(fig1, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/time_step", Figure(fig2, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/position", Figure(fig3, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/velocity", Figure(fig4, close = True), exclude=("stdout", "log", "json", "csv"))
        
        # Save deltaV and time to Tensorboard
        time = t[-1]
        deltaV = info[-1].get('deltaV')
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
        self.env = self.model.get_env()

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
           "initial std (log)": self.model.policy_kwargs.get('log_std_init'),
           "u max": self.env.get_attr('u_max')[0],
           "variance type": self.env.get_attr('variance_type')[0],
           "variance value": self.env.get_attr('variance')[0],
           "max duration": self.env.get_attr('max_duration')[0],
           "map limits": np.array2string(self.env.get_attr('map_limits')[0]),
           "time step limits": np.array2string(self.env.get_attr('t_step_limits')[0]),
           "initial state": np.array2string(self.env.get_attr('initial_state')[0]),
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
    
class TextDataCallback(BaseCallback):
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:

        output_formats = self.logger.output_formats
        self.tb_writer = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat)).writer
        
        self.env = self.model.get_env()

        np.printoptions(linewidth=1000)
        
        # Log Hparams in Text
        hparam_text = \
             "Algorithm: " + str(self.model.__class__.__name__) + "<br>" +\
             "Learning Rate: " + str(self.model.learning_rate) + "<br>" +\
             "N Steps: " + str(self.model.n_steps) + "<br>" +\
             "N Epochs: " + str(self.model.n_epochs) + "<br>" +\
             "Batch Size: " + str(self.model.batch_size) + "<br>" +\
             "Gamma: " + str(self.model.gamma) + "<br>" +\
             "Entropy Coefficient: " + str(self.model.ent_coef) + "<br>" +\
             "Value Function Coefficient: " + str(self.model.vf_coef) + "<br>" +\
             "Max Gradient Norm: " + str(self.model.max_grad_norm) + "<br>" +\
             "Initial Std (log): " + str(self.model.policy_kwargs.get('log_std_init')) + "<br>"

        self.tb_writer.add_text("hparams", hparam_text)
        
        # Log environment parameters
        env_text = \
            "Variance Type: " + str(self.env.get_attr('variance_type')[0]) + "<br>"\
            "Variance Value: " + str(self.env.get_attr('variance')[0]) + "<br>"\
            "Max Duration: " + str(self.env.get_attr('max_duration')[0]) + "<br>"\
            "U Max: " + str(self.env.get_attr('u_max')[0]) + "<br>"\
            "Map Limits: " + np.array2string(self.env.get_attr('map_limits')[0]) + "<br>"\
            "Time Step Limits: " + np.array2string(self.env.get_attr('t_step_limits')[0]) + "<br>"\
            "Initial State" + np.array2string(self.env.get_attr('initial_state')[0]) + "<br>"\
            "Max Pos: " + str(self.env.get_attr('max_pos')[0]) + "<br>"\
            "Max Vel: " + str(self.env.get_attr('max_vel')[0]) + "<br>"

        self.tb_writer.add_text("environment", env_text)

        # Log Reward Function
        self.tb_writer.add_text("reward", self.extract_data("SpacecraftEnv.py"))
        
        self.tb_writer.flush()

    def _on_step(self) -> bool:
        return True
    
    def extract_data(self, filename):
        lines = []
        file = open("SpacecraftEnv.py", "rt")
        for line in file:
            lines.append(line)
        file.close()

        printing = False
        output = "#---------- <br>"

        for line in lines:
            if(line.find("#$$") != -1):
                printing = not printing
                
            if(printing):
                output = output + line

        output = output + "#---------- <br>"
        output = output.replace("$$", "")
        output = output.replace("\n", "<br>")
        return output