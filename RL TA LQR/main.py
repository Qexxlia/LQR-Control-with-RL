from stable_baselines3 import A2C 
from stable_baselines3 import PPO 
from SpacecraftEnv import SpacecraftEnv as spe
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList

from Callbacks import TensorBoardCallback

import SpacecraftDynamics as scd
import time


# scd.printAMatrice(7500, 3.986004418E5)

env = spe()
obs = env.reset()

timeStr = time.strftime("%Y%m%d-%H%M%S")

eval_callback = EvalCallback(env, best_model_save_path="./models/Spacecraft/A2C/" + timeStr + "/best_model/", log_path="./models/Spacecraft/A2C/" + timeStr + "/evaluations/", eval_freq=25, deterministic=True, render=False, verbose=1)
# tensorboard_callback = TensorBoardCallback(env)

callbacks = CallbackList([eval_callback])

# model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./models/Spacecraft/A2C/logs/")
model = A2C.load("./models/Spacecraft/A2C/20240507-164406 - SI/best_model/best_model.zip", env=env, tensorboard_log="./models/Spacecraft/A2C/logs/", verbose = 1)

model.learn(total_timesteps=10000, progress_bar=True, callback=callbacks)

# vec_env = model.get_env()
# obs = vec_env.reset()

# model.predict(obs, deterministic=True)
# obs = env.step(model.predict(obs, deterministic=True)[0])