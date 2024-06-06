import time
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np

from SpacecraftEnv import SpacecraftEnv as spe
from Callbacks import StateCallback
from Callbacks import HParamCallback

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
variance_percentage = 0.00
max_duration = 5000
map_limits = np.array(
    [
        [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
        [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
    ],
    dtype=np.float32
)
tStep = 5
u_max = 1e-4
verbose = 0

device = 'cpu' #cpu or cuda

# NUM EPISODES
num_episodes = 250 

# TEST TYPE
testtype = "u_max"
additional_info = "__u_max=" + str(u_max)

timeStr = "20240605-1235"

nameStr = "./models/spacecraft/" + testtype + "/PPO_" + timeStr + additional_info

num_time_steps = num_episodes * n_steps

# Create environment
env = spe(verbose=verbose, variance_percentage=variance_percentage, maxDuration=max_duration, map_limits=map_limits, tStep=tStep, u_max=u_max)
env = Monitor(env)
obs = env.reset()

# Define Callbacks
eval_callback = EvalCallback(env, best_model_save_path=nameStr + "/best_model/", log_path=nameStr + "/evaluations/", eval_freq=25*n_steps, deterministic=True, render=False, verbose=0)
state_callback = StateCallback(csv_save_path=nameStr + "/data/", map_limits=map_limits, max_duration=max_duration)
hparam_callback = HParamCallback()
callbacks = CallbackList([state_callback, hparam_callback, eval_callback])

# Define PPO Parameters
policy_kwargs = {
    'share_features_extractor' : False,
    'log_std_init' : log_std_init,
}

model = PPO.load(nameStr + "/best_model/best_model.zip", 
                num_time_steps=256000, 
                env=env, 
                device=device)

# Start learning
model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks, reset_num_timesteps=False)