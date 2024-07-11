import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sbx import PPO
from drone_env import DroneEnv as de

# Environment Parameters
env_params = {
    "variance_type" : "none",
    "variance" : 0,
    "max_duration" : 750,
    "map_limits" : np.array(
        [
            np.ones(16)*0.0001,
            np.ones(16)*500
        ],
        dtype=np.float32
    ),
    "t_step_limits" : np.array([0.1, 10], dtype=np.float32),
    "u_max" : 24,
    "seed" : 0,
    "absolute_norm" : False
}

env = de(verbose=False, args=env_params)

model = PPO.load(os.getcwd() +"/models/drone/PPO_20240702-1652/final_model/final_model.zip", env=env, )

env.set_episode_options({'deterministic': 1, 'verbose': 0})
[obs, _nfo] = env.reset()

[action, _state] = model.predict(obs, deterministic=True)
action_record = action

done = False
while not done:
    [obs, reward, done, done2, info] = env.step(action)
    [action, _state] = model.predict(obs, deterministic=True)
    action_record = np.vstack((action_record, action))

# Extract data
attitude = info.get('attitude')
spin = info.get('spin')
settling_cost = info.get('settling_cost')
overshoot_cost = info.get('overshoot_cost')
t = info.get('t')
u = info.get('u')

## PLOT DATA
t_e = np.vstack((t,t,t)).transpose()
t_e1 = np.vstack((t,t,t,t)).transpose()

# Font sizes
tsL = 12
tsS = 7

def create_plot(fig_num, title, xlabel, ylabel, x_data, y_data, labels):
    fig = plt.figure(fig_num, figsize=(10,2)) 
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_data, y_data, label=labels)
    ax.set_title(title, fontsize=tsL)
    ax.set_xlabel(xlabel, fontsize=tsS)
    ax.set_ylabel(ylabel, fontsize=tsS)
    if(labels != None):
        fig.legend()
    return fig

# Voltage
fig3 = create_plot(3, 'Voltage', 'Time (s)', 'Voltage (V)', t_e1, u.transpose(), ['F','B','L','R'])

# Attitude
fig4 = create_plot(5, 'Attitude', 'Time (s)', 'Angle (rad)', t_e, attitude.transpose(), ['roll','pitch', 'yaw'])

# Spin
fig6 = create_plot(7, 'Spin', 'Time (s)', 'Spin (rad/s)', t_e, spin.transpose(), ['roll','pitch','yaw'])

r_overshoot = (np.max(attitude[0,:]) - env.desired_angle)/env.desired_angle
p_overshoot = (np.max(attitude[1,:]) - env.desired_angle)/env.desired_angle
y_overshoot = (np.max(attitude[2,:]) - env.desired_angle)/env.desired_angle
overshoot_cost = math.sqrt(r_overshoot**2 + p_overshoot**2 + y_overshoot**2)

print("Roll Overshoot: " + str(r_overshoot))
print("Pitch Overshoot: " + str(p_overshoot))
print("Yaw Overshoot: " + str(y_overshoot))

def calc_settling(time, data):
    settling_time = 0
    for i in range(len(data)-1, 0, -1):
        if abs(data[i]) >= 0.02*env.desired_angle:
            settling_time = time[i-1]
            break
    return settling_time

r_settling = calc_settling(t, attitude[0,:])
p_settling = calc_settling(t, attitude[1,:])
y_settling = calc_settling(t, attitude[2,:])

print("Roll Settling Time: " + str(r_settling))
print("Pitch Settling Time: " + str(p_settling))
print("Yaw Settling Time: " + str(y_settling))

plt.show()