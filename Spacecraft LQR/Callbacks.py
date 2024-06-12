from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, HParam, TensorBoardOutputFormat

import SpacecraftDynamics as scd

class StateCallback(BaseCallback):
    def __init__(self, args):
        super().__init__(0)

        self.map_limits = args.get('map_limits')
        self.max_duration = args.get('max_duration')
        self.t_step_limits = args.get('t_step_limits')
        

    def _on_rollout_end(self):
        # Get environment
        vec_env = self.model.get_env()
        
        # Reset environment for variance-free simulation
        vec_env.env_method("set_episode_options", {'deterministic': 1, 'verbose': 0})
        obs = vec_env.reset()
            
        # Initial Action
        [action, _state] = self.model.predict(obs, deterministic=True)

        # Record all actions taken
        action_record = action

        # Simulate the spacecraft
        done = False
        while not done:
            [obs, reward, done, info] = vec_env.step(action)
            [action, _state] = self.model.predict(obs, deterministic=True)

            action_record = np.vstack((action_record, action))

            if(info[-1].get('t')[-1] > self.max_duration):
                done = True

        # Extract data
        pos = info[-1].get('pos')
        vel = info[-1].get('vel')
        t = info[-1].get('t')
        reward = reward[-1]

        qWeights = vec_env.env_method("map_range", action_record[:,0:6], -1, 1, self.map_limits[0,0:6], self.map_limits[1,0:6])[0]
        rWeights = vec_env.env_method("map_range", action_record[:,6:9], -1, 1, self.map_limits[0,6:9], self.map_limits[1,6:9])[0]
        t_step = vec_env.env_method("map_range", action_record[:,9], -1, 1, self.t_step_limits[0], self.t_step_limits[1])[0]

        ## PLOT DATA

        # Action time steps
        first = True
        for step in t_step:
            if first:
                t_a = np.array([step])
                first = False
            else:
                t_a = np.append(t_a, t_a[-1]+step)
        
        # Time arrays for plotting
        t_a1 = np.vstack((t_a, t_a, t_a)).transpose()
        t_a2 = t_a
        t_a = np.vstack((t_a, t_a, t_a, t_a, t_a, t_a)).transpose()

        t_e = np.vstack((t,t,t)).transpose()

        # Font sizes
        tsL = 12
        tsS = 7

        def create_plot(fig_num, title, xlabel, ylabel, x_data, y_data, labels):
            fig = plt.figure(fig_num, figsize=(10,2)) 
            ax = fig.add_subplot(1,1,1)
            ax.plot(x_data, y_data, label=labels)
            ax.set_title(title, fontsize=tsL)
            ax.set_xlabel(xlabel, fontsize=tsS)
            ax.set_ylabel(ylabel, fontsize=tsS)
            fig.legend()
            return fig
        
        # Q Weights
        fig = create_plot(1, 'Q Weights', 'Time (s)', 'Weight', t_a, qWeights, ['Q0','Q1','Q2','Q3','Q4','Q5'])
        
        # R Weights
        fig1 = create_plot(2, 'R Weights', 'Time (s)', 'Weight', t_a1, rWeights, ['R0','R1','R2'])
        
        # Time Step
        fig2 = create_plot(3, 'Time Step', 'Time (s)', 'Time Step (s)', t_a2, t_step, None)
        
        # Position
        fig3 = create_plot(4, 'Position', 'Time (s)', 'Position (km)', t_e, pos[0:3, :].transpose(), ['x','y','z'])
        
        # Velocity
        fig4 = create_plot(5, 'Velocity', 'Time (s)', 'Velocity (km/s)', t_e, vel[0:3, :].transpose(), ['x','y','z'])
        
        # Log plots to tensorboard
        self.logger.record("plots/action/q_weights", Figure(fig, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/action/r_weights", Figure(fig1, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/action/time_step", Figure(fig2, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/state/position", Figure(fig3, close = True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("plots/state/velocity", Figure(fig4, close = True), exclude=("stdout", "log", "json", "csv"))

        ## LOG DATA
        self.logger.record("time/ep_time_elapsed", t[-1], exclude=("stdout", "log", "json", "csv"))

        data = ['deltaV_punishment', 'velocity_punishment', 'distance_punishment', 'time_punishment', 'truncated_punishment', 'reward']
        for item in data:
            value = info[-1].get(item)
            self.logger.record(f"reward/{item}", value, exclude=("stdout", "log", "json", "csv"))
    
        return True
    

    def _on_step(self):
        return True



class HParamCallback(BaseCallback):
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self.env = self.model.get_env()

        # Parameters to log
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
            "Max Pos: " + str(self.env.get_attr('max_pos')[0]) + "<br>"\
            "Max Vel: " + str(self.env.get_attr('max_vel')[0]) + "<br>"

        self.tb_writer.add_text("environment", env_text)

        # Log Reward Function    
        def extract_data(filename):
            lines = []
            file = open("SpacecraftEnv.py", "rt")
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

        self.tb_writer.add_text("reward", extract_data("SpacecraftEnv.py"))
        
        self.tb_writer.flush()

    def _on_step(self) -> bool:
        return True
    
