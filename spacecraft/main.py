import time
# from stable_baselines3 import PPO 
from sbx import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import numpy as np
from math import pi
import os
import tkinter
tkinter.Tk().withdraw()
from tkinter import filedialog

from spacecraft_env import SpacecraftEnv as spe
from callbacks import StateCallback
from callbacks import HParamCallback
from callbacks import TextDataCallback


#-------------------- PARAMETERS --------------------
# Hyperparameters
learning_rate = 1e-4
n_steps = 2048 
batch_size = 64
n_epochs = 10
clip_range = 0.1
gamma = 0.99

vf_coef = 0.5
ent_coef = 0.0
gae_lambda = 0.95
max_grad_norm = 0.5
seed = 0

log_std_init = np.log(0.3)

# Environment Parameters
env_params = {
    "variance_type" : "range",
    "variance" : 0,
    "max_duration" : 750,
    "map_limits" : np.array(
        [
            [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
            [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
        ],
        dtype=np.float32
    ),
    "t_step_limits" : np.array([1, 100], dtype=np.float32),
    "u_max" : 1e-3,
    "simulation_type" : "qrt",
    "t_step" : 0,
    "seed" : seed,
    "absolute_norm" : False
}

verbose = 0
device = 'cpu' #cpu or cuda

# NUM EPISODES
num_episodes = 1000

# TEST TYPE
testtype = "abs_norm"
additional_info = ""

#-------------------- INITIALISE MODEL & RUN TRAINING --------------------

timeStr = time.strftime("%Y%m%d-%H%M")
if(input("Run as test? [Y/n]: ") != "n"):
    name_string = "./models/testing/spacecraft/" + "PPO_" + timeStr + additional_info
    print("TESTING")
else:
    name_string = "./models/spacecraft/" + testtype + "/PPO_" + timeStr + additional_info
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

num_time_steps = num_episodes * n_steps

# Create environment
env = spe(verbose=verbose, args=env_params)
env = Monitor(env)
obs = env.reset()

# Define Callbacks
eval_callback = EvalCallback(env, best_model_save_path=name_string + "/best_model/", log_path=name_string + "/evaluations/", eval_freq=5*n_steps, deterministic=True, render=False, verbose=0)
state_callback = StateCallback(args=env_params, plot_rate = 5)
hparam_callback = HParamCallback()
text_data_callback = TextDataCallback()
checkpoint_callback = CheckpointCallback(save_freq=50*n_steps, save_path=name_string + "/checkpoints/")
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
    model = PPO.load(name_string + "/best_model/best_model.zip", 
                    env=env, 
                    device=device)

    # Start learning
    model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks, reset_num_timesteps=False)
    

model.save(name_string + "/final_model/final_model.zip")