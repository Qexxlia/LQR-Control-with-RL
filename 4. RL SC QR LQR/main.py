import time
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
import numpy as np

from SpacecraftEnv import SpacecraftEnv as spe
from Callback import PlotCallback
import SpacecraftDynamics as scd


np.set_printoptions(linewidth=np.inf)
# sol = scd.simulate([0.5, -0.5, 0, 1e-3, -1e-3, 0, 30], (0, 50), np.ones(6), np.ones(3))
# print(sol.y[:,-1])
# print(sol.t[-1])
# scd.printAMatrice(6371 + 500, 3.986e5)
# time.sleep(100)

#-------------------- SETUP --------------------
Testing = True

if input("Run as a test? (Y/n) : ") == 'n':
    Testing = False

env = spe()
obs = env.reset()

timeStr = time.strftime("%Y%m%d-%H%M")

#-------------------- HYPERPARAMETERS --------------------

learning_rate = 3e-4
n_steps = 1024 # 1024 seems to be the best balance
batch_size = 64
n_epochs = 10
clip_range = 0.2
gamma = 0.99

vf_coef = 0.5
ent_coef = 0.0
gae_lambda = 0.95
max_grad_norm = 0.5

log_std_init = np.log(1)

#-------------------- TO CHANGE --------------------
device = 'cpu' #cpu or cuda

# NUM EPISODES
num_episodes = 50 

# TEST TYPE
testtype = "state_variance"

additional_info = "__var-scale=5e-2_5e-5"

#-------------------- TRAINING --------------------
num_time_steps = num_episodes * n_steps
if Testing:
    nameStr = "./models/testing/spacecraft/" + "PPO_" + timeStr + "__LR=" + str(learning_rate) + "__NS=" + str(n_steps) + "__BS=" + str(batch_size) + "__NE=" + str(n_epochs) + "__CR=" + str(clip_range) + "__G=" + str(gamma) + "__LSI=" + str(log_std_init) + additional_info
    print("TESTING MODE")
else:
    nameStr = "./models/spacecraft/" + testtype + "/PPO_" + timeStr + "__LR=" + str(learning_rate) + "__NS=" + str(n_steps) + "__BS=" + str(batch_size) + "__NE=" + str(n_epochs) + "__CR=" + str(clip_range) + "__G=" + str(gamma) + "__LSI=" + str(log_std_init) + additional_info

print("Saving to: " + nameStr + "\n")

eval_callback = EvalCallback(env, best_model_save_path=nameStr + "/best_model/", log_path=nameStr + "/evaluations/", eval_freq=200, deterministic=True, render=False, verbose=0)
plot_callback = PlotCallback(verbose=0, csv_save_path=nameStr + "/data/")

callbacks = CallbackList([eval_callback, plot_callback])

policy_kwargs = {
    'share_features_extractor' : False,
    'log_std_init' : log_std_init,
}

model = PPO(
    "MlpPolicy", 
    env, 
    tensorboard_log=nameStr + "/logs/",
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

# model = A2C.load("./models/Spacecraft/A2C/20240507-164406 - SI/best_model/best_model.zip", env=env, tensorboard_log="./models/Spacecraft/A2C/logs/", verbose = 1)

model.learn(total_timesteps=num_time_steps, progress_bar=True, callback=callbacks)

# vec_env = model.get_env()
# obs = vec_env.reset()

# model.predict(obs, deterministic=True)
# obs = env.step(model.predict(obs, deterministic=True)[0])
