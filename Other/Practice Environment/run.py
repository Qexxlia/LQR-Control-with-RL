import gymnasium
import time
from stable_baselines3 import A2C

import tensorflow as tf

from env import testEnv

env = testEnv()


obs = env.reset()

# for _ in range(10000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation = env.reset()
#         time.sleep(1000)



model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_test_tensorboard/")

model.learn(total_timesteps=100000,progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
dones = False 
while (not dones):
   action, _states = model.predict(obs)
   obs, rewards, dones, info = vec_env.step(action)