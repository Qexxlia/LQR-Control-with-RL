from typing import Dict, Any
import os

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, HParam, TensorBoardOutputFormat

import drone_dynamics as dd

class StateCallback(BaseCallback):
    def __init__(self, args, plot_rate):
        super().__init__(0)

        self.map_limits = args.get('map_limits')
        self.max_duration = args.get('max_duration')
        self.t_step_limits = args.get('t_step_limits')
        self.plot_rate = plot_rate
        self.rollouts = 0
        

    def _on_rollout_end(self):
        ## PLOT DATA
        if(self.rollouts % self.plot_rate == 0):
            # Get environment
            vec_env = self.model.get_env()
            
            # Reset environment for variance-free simulation
            vec_env.env_method("set_episode_options", {'deterministic': 1, 'verbose': 0})
            obs = vec_env.reset()
                
            # Initial Action
            [action, _state] = self.model.predict(obs, deterministic=True)

            # Record all actions taken
            action_record = action

            # Simulate
            done = False
            while not done:
                [obs, reward, done, info] = vec_env.step(action)
                [action, _state] = self.model.predict(obs, deterministic=True)

                action_record = np.vstack((action_record, action))

                if(info[-1].get('t')[-1] > self.max_duration):
                    done = True
                    
            q_weights = np.zeros((np.shape(action_record)[0], 12))
            r_weights = np.zeros((np.shape(action_record)[0], 2))

            for i in range(0, np.shape(action_record)[0]):
                Q_ang = vec_env.env_method("log_map_range", action_record[i,0:6], -1, 1, np.ones(6)*self.map_limits[0,0], np.ones(6)*self.map_limits[1,0])
                R_ang = vec_env.env_method("log_map_range", action_record[i,6], -1, 1, self.map_limits[0,1], self.map_limits[1,1])
                Q_pos = vec_env.env_method("log_map_range", action_record[i,7:13], -1, 1, np.ones(6)*self.map_limits[0,2], np.ones(6)*self.map_limits[1,2])
                R_pos = vec_env.env_method("log_map_range", action_record[i,13], -1, 1, self.map_limits[0,3], self.map_limits[1,3])
                
                q_weights[i] = np.hstack((Q_ang, Q_pos))
                r_weights[i] = np.hstack((R_ang, R_pos))

            # Extract data
            attitude = info[-1].get('attitude')
            position = info[-1].get('position')
            t = info[-1].get('t')
            reward = reward[-1]
            t_step = self.t_step_limits[0]


            t_a = np.array([0, t_step])
            while np.shape(t_a)[0] < np.shape(q_weights)[0]:
                t_a = np.append(t_a, t_a[-1] + t_step)
                
            t_step = np.ones(np.shape(t_a)) * t_step
            
            # Time arrays for plotting
            t_a1 = np.vstack((t_a, t_a)).transpose()

            t_a2 = t_a
            t_a = np.vstack((t_a, t_a, t_a, t_a, t_a, t_a, t_a, t_a, t_a, t_a, t_a, t_a)).transpose()

            t_e = np.vstack((t,t,t)).transpose()

            # Font sizes
            tsL = 12
            tsS = 7

            def create_plot(fig_num, title, xlabel, ylabel, x_data, y_data, labels, log):
                fig = plt.figure(fig_num, figsize=(10,2)) 
                ax = fig.add_subplot(1,1,1)
                ax.plot(x_data, y_data, label=labels)
                ax.set_title(title, fontsize=tsL)
                ax.set_xlabel(xlabel, fontsize=tsS)
                ax.set_ylabel(ylabel, fontsize=tsS)
                if log:
                    ax.set_yscale('log')
                if(labels != None):
                    fig.legend()
                return fig
            
            # Q Weights
            fig = create_plot(1, 'Q Weights', 'Time (s)', 'Weight', t_a, q_weights, ['A0','A1','A2','A3','A4','A5','P0','P1','P2','P3','P4','P5'], True)
            
            # R Weights
            fig1 = create_plot(2, 'R Weights', 'Time (s)', 'Weight', t_a1, r_weights, ['A','P'], True)
            
            # Time Step
            fig2 = create_plot(3, 'Time Step', 'Time (s)', 'Time Step (s)', t_a2, t_step, None, False)
            
            # Attitude
            fig4 = create_plot(5, 'Attitude', 'Time (s)', 'Angle (rad)', t_e, attitude.transpose(), ['roll','pitch', 'yaw'], False)

            # Spin
            fig6 = create_plot(7, 'Position', 'Time (s)', 'Spin (rad/s)', t_e, position.transpose(), ['x','y','z'], False)
            
            # Log plots to tensorboard
            self.logger.record("plots/action/q_weights", Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))
            self.logger.record("plots/action/r_weights", Figure(fig1, close = True), exclude=("stdout", "log", "json", "csv"))
            self.logger.record("plots/action/time_step", Figure(fig2, close = True), exclude=("stdout", "log", "json", "csv"))
            self.logger.record("plots/state/attitude", Figure(fig4, close = True), exclude=("stdout", "log", "json", "csv"))
            self.logger.record("plots/state/position", Figure(fig6, close = True), exclude=("stdout", "log", "json", "csv"))

            ## LOG DATA
            self.logger.record("time/ep_time_elapsed", t[-1], exclude=("stdout", "log", "json", "csv"))
            self.logger.record("reward/reward", reward, exclude=("stdout", "log", "json", "csv"))
            
        self.rollouts += 1
        return True
        

    def _on_step(self):
        return True



class HParamCallback(BaseCallback):
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self.env = self.model.get_env()

        # Parameters to log
        hparam_dict = {
           "algorithm": self.model.__class__.__name__,
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
           "absolute norm": self.env.get_attr('absolute_norm')[0],
           "seed": self.env.get_attr('seed')[0],
        }

        # Metrics to log
        metric_dict = {
            "train/clip_range": 0,
        }

        # Log hparams and metrics
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
        

    def _on_step(self) -> bool:
        return True


    
class TextDataCallback(BaseCallback):
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:

        # Get TB writer
        output_formats = self.logger.output_formats
        self.tb_writer = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat)).writer
        
        # Get environment
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
            "Absolute Norm: " + str(self.env.get_attr('absolute_norm')[0]) + "<br>" +\
            "Seed: " + str(self.env.get_attr('seed')[0]) + "<br>"

        self.tb_writer.add_text("environment", env_text)

        # Log Reward Function    
        def extract_data(filename):
            lines = []
            file = open(filename, "rt")
            for line in file:
                lines.append(line)
            file.close()

            printing = False
            output = ""

            for line in lines:
                if(line.find("#$$") != -1):
                    printing = not printing
                    
                if(printing):
                    output = output + line

            output = output.replace("$$", "")
            output = output.replace("\n", "<br>")
            return output

        self.tb_writer.add_text("reward", extract_data("./drone_env.py"))
        
        self.tb_writer.flush()

    def _on_step(self) -> bool:
        return True
    

class UMaxCallback(BaseCallback):
    def __init__(self, args):
        super().__init__(0)
        self.initial_u_max = args.get('u_max_initial')
        self.final_u_max = args.get('u_max_final')
        self.step_gap = args.get('step_gap')
        self.i = 0
        self.total_timesteps = args.get('total_timesteps')
        self.u_max_list = np.linspace(self.initial_u_max, self.final_u_max, (int)(self.total_timesteps/self.step_gap))
        self.env = None

    def _on_step(self) -> bool:
        if(self.num_timesteps % self.step_gap == 0):
            self.env = self.training_env
            self.env.env_method("update_u_max", self.u_max_list[self.i])
            self.i += 1

        self.logger.record("other/u_max", self.u_max_list[self.i], exclude=("stdout", "log", "json", "csv"))
        return True