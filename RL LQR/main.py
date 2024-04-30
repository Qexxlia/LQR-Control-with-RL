from stable_baselines3 import a2c
from SpacecraftEnv import SpacecraftEnv as spe
from stable_baselines3.common.callbacks import CheckpointCallback

import SpacecraftDynamics as scd

# scd.printAMatrice(7500, 3.986004418E5)

env = spe()
obs = env.reset()

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/Spacecraft/A2C/checkpoints/", name_prefix="model", verbose=2)
model = a2c.A2C("MlpPolicy", env, verbose=1, tensorboard_log="./models/Spacecraft/A2C/logs/")

model.learn(total_timesteps=10000, progress_bar=True, callback=checkpoint_callback)

vec_env = model.get_env()
obs = vec_env.reset()