import time
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
from math import pi

from SpacecraftEnv import SpacecraftEnv as spe
from Callbacks import StateCallback
from Callbacks import HParamCallback
from Callbacks import TextDataCallback

#-------------------- PARAMETERS --------------------
# Hyperparameters
learning_rate = 3e-2
n_steps = 1024
batch_size = 64
n_epochs = 10
clip_range = 0.2
gamma = 0.99

vf_coef = 0.5
ent_coef = 0.0
gae_lambda = 0.95
max_grad_norm = 0.5

log_std_init = np.log(1)

# Environment Parameters
env_params = {
    "variance_type" : "none",
    "variance" : pi/2,
    "max_duration" : 1000,
    "map_limits" : np.array(
        [
            [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
            [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
        ],
        dtype=np.float32
    ),
    "t_step_limits" : np.array([1e-6, 1e1], dtype=np.float32),
    "u_max" : 1e-3
}

verbose = 0
device = 'cuda' #cpu or cuda

# NUM EPISODES
num_episodes = 750 

# TEST TYPE
testtype = "TA"
additional_info = ""

timeStr = "20240612-0118"

nameStr = "./models/spacecraft/" + testtype + "/PPO_" + timeStr + additional_info

num_time_steps = num_episodes * n_steps
prev_time_steps = 51200

# Create environment
env = spe(verbose=verbose, args=env_params)
env = Monitor(env)
obs = env.reset()

# Define Callbacks
eval_callback = EvalCallback(env, best_model_save_path=nameStr + "/best_model/", log_path=nameStr + "/evaluations/", eval_freq=25*n_steps, deterministic=True, render=False, verbose=0)
state_callback = StateCallback(args=env_params)
hparam_callback = HParamCallback()
text_data_callback = TextDataCallback()
callbacks = CallbackList([state_callback, hparam_callback, eval_callback, text_data_callback])

# Define PPO Parameters
policy_kwargs = {
    'share_features_extractor' : False,
    'log_std_init' : log_std_init,
}

model = PPO.load(nameStr + "/best_model/best_model.zip", 
                num_time_steps=num_time_steps, 
                env=env, 
                device=device)

# Start learning
model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks, reset_num_timesteps=False)