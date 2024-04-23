from stable_baselines3 import a2c
from SpacecraftEnv import SpacecraftEnv as spe

env = spe()

obs = env.reset()

model = a2c.A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tensorflow/Spacecraft/A2C")

model.learn(total_timesteps=1000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()

terminated = False
truncated = False

while not terminated and not truncated:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = vec_env.step(action)