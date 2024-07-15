import time
from stable_baselines3 import PPO 
# from sbx import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from typing import Callable

import numpy as np
from math import pi
import os
import tkinter
from tkinter import filedialog

from drone_env import DroneEnv as de
from callbacks import StateCallback
from callbacks import HParamCallback
from callbacks import TextDataCallback
from callbacks import UMaxCallback

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

#-------------------- PARAMETERS --------------------
# Hyperparameters
learning_rate = linear_schedule(1e-3)
n_steps = 1024
batch_size = 64
n_epochs = 10
clip_range = 0.1
gamma = 0.99

vf_coef = 0.5
ent_coef = 0.0
gae_lambda = 0.95
max_grad_norm = 0.5
seed = 0

log_std_init = np.log(1)

# NUM EPISODES
num_episodes = 1500
num_time_steps = num_episodes * n_steps

# Environment Parameters
env_params = {
    "variance_type" : "none",
    "variance" : 0,
    "max_duration" : 50,
    "map_limits" : np.array(
        [
            np.ones(4)*1,
            np.ones(4)*1000
        ],
        dtype=np.float32
    ),
    "t_step_limits" : np.array([1], dtype=np.float32),
    "u_max" : 24,
    "seed" : seed,
    "absolute_norm" : False
}

""" u_max_callback_args = {
    "u_max_initial" : env_params["u_max"],
    "u_max_final" : 24,
    "step_gap" : 1024*100,
    "total_timesteps" : num_time_steps,
} """

verbose = 0
device = 'cpu' #cpu or cuda


# TEST TYPE
testtype = ""
additional_info = ""

#-------------------- INITIALISE MODEL & RUN TRAINING --------------------

timeStr = time.strftime("%Y%m%d-%H%M")
a = input("Run as test? [Y/n]: ")
# if(a == "pso"):
#     print("PSO")
#     pso = dpso(env_params, verbose)
#     cost, pos = pso.optimize()
#     print("PSO Complete")
#     print(cost)
#     print(pos)
#     exit()
if(a != "n"):
    name_string = "./models/testing/drone/" + "PPO_" + timeStr + additional_info
    print("TESTING")
else:
    name_string = "./models/drone/" + testtype + "/PPO_" + timeStr + additional_info
    i = 1
    while os.path.exists(name_string):
        name_string = name_string + "_" + str(i)
        i = i + 1
    
if(input("Continue training? [y/N]: ") == "y"):
    continue_training = True
    name_string = filedialog.askdirectory(initialdir=os.getcwd()+"/models")
else:
    continue_training = False

print("Path: " + name_string)

# Create environment
env = de(verbose=verbose, args=env_params)
env = Monitor(env)
obs = env.reset()

# Define Callbacks
eval_callback = EvalCallback(env, best_model_save_path=name_string + "/best_model/", log_path=name_string + "/evaluations/", eval_freq=5*n_steps, deterministic=True, render=False, verbose=0)
state_callback = StateCallback(args=env_params, plot_rate = 5)
hparam_callback = HParamCallback()
text_data_callback = TextDataCallback()
checkpoint_callback = CheckpointCallback(save_freq=50*n_steps, save_path=name_string + "/checkpoints/")
# u_max_callback = UMaxCallback(args=u_max_callback_args)
callbacks = CallbackList([state_callback, hparam_callback, eval_callback, text_data_callback, checkpoint_callback])

# Define PPO Parameters
policy_kwargs = {
    'share_features_extractor' : False,
    'log_std_init' : log_std_init,
}

if not continue_training:
    model = PPO(
        "MlpPolicy", 
        env, 
        tensorboard_log=name_string + "/logs/",
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_steps=n_steps,
        clip_range=clip_range,
        device=device,
        policy_kwargs=policy_kwargs,
        seed = seed
    )

    # Start learning
    model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks)
else:
    model = PPO.load(name_string + "/final_model/final_model.zip", 
                    env=env, 
                    device=device,
                    tensorboard_log=name_string + "/logs/"
                    )

    # Start learning
    model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks, reset_num_timesteps=False)
    

model.save(name_string + "/final_model/final_model.zip")