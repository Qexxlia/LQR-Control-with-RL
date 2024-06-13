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
learning_rate = 1e-4
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
    "variance_type" : "range",
    "variance" : pi/8,
    "max_duration" : 750,
    "map_limits" : np.array(
        [
            [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
            [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
        ],
        dtype=np.float32
    ),
    "t_step_limits" : np.array([1, 100], dtype=np.float32),
    "u_max" : 1e-3
}

verbose = 0
device = 'cpu' #cpu or cuda

# NUM EPISODES
num_episodes = 250

# TEST TYPE
testtype = "TA"
additional_info = ""

#-------------------- INITIALISE MODEL & RUN TRAINING --------------------

if(input("Run as test? [Y/n]: ") != "n"):
    testing = True
else:
    testing = False

timeStr = time.strftime("%Y%m%d-%H%M")

# Constant calculations/definitions
if testing:
    name_string = "./models/testing/spacecraft/" + "PPO_" + timeStr + additional_info
    print("TESTING MODE")
else:
    name_string = "./models/spacecraft/" + testtype + "/PPO_" + timeStr + additional_info
print("Saving to: " + name_string)

num_time_steps = num_episodes * n_steps

# Create environment
env = spe(verbose=verbose, args=env_params)
env = Monitor(env)
obs = env.reset()

# Define Callbacks
eval_callback = EvalCallback(env, best_model_save_path=name_string + "/best_model/", log_path=name_string + "/evaluations/", eval_freq=25*n_steps, deterministic=True, render=False, verbose=0)
state_callback = StateCallback(args=env_params)
hparam_callback = HParamCallback()
text_data_callback = TextDataCallback()
callbacks = CallbackList([state_callback, hparam_callback, eval_callback, text_data_callback])

# Define PPO Parameters
policy_kwargs = {
    'share_features_extractor' : False,
    'log_std_init' : log_std_init,
}

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
    policy_kwargs=policy_kwargs
)

# Start learning
model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks)