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
tsS = 10

# Voltage
# fig3 = create_plot(3, 'Voltage', 'Time (s)', 'Voltage (V)', t_e1, u.transpose(), ['F','B','L','R'])


r_overshoot = (np.max(attitude[0,:]) - env.desired_angle)/env.desired_angle
p_overshoot = (np.max(attitude[1,:]) - env.desired_angle)/env.desired_angle
y_overshoot = (np.max(attitude[2,:]) - env.desired_angle)/env.desired_angle
overshoot_cost = math.sqrt(r_overshoot**2 + p_overshoot**2 + y_overshoot**2)

print("Roll Overshoot: " + str(100*r_overshoot))
print("Pitch Overshoot: " + str(100*p_overshoot))
print("Yaw Overshoot: " + str(100*y_overshoot))

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

# Attitude
# fig4 = create_plot(5, 'Attitude', 'Time (s)', 'Angle (rad)', t_e, attitude.transpose(), ['roll','pitch', 'yaw'])

# Spin
# fig6 = create_plot(7, 'Spin', 'Time (s)', 'Spin (rad/s)', t_e, spin.transpose(), ['roll','pitch','yaw'])


fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Quadcopter Attitude and Spin', fontsize=tsL)
line = ax1.plot(t_e, np.degrees(attitude.transpose()), label=['roll','pitch', 'yaw'])
line[0].set_dashes([2,2,10,2])
line[0].set_dash_capstyle('round')
line[1].set_dashes([5,2,5,2])
line[1].set_dash_capstyle('round')
line[2].set_dashes([2,2,2,2])
line[2].set_dash_capstyle('round')
ax1.axvspan(0, 0.1, color='lightgrey', alpha=0.5)
ax1.axvspan(0.2, 0.3, color='lightgrey', alpha=0.5)
ax1.axvspan(0.4, t[-1], color='lightgrey', alpha=0.5)
ax1.hlines(5, 0, t[-1], colors='grey', linestyles='dashed')
ax1.set_ylabel('Attitude (deg)', fontsize=tsS)

line = ax2.plot(t_e, np.degrees(spin.transpose()), label=['_roll','_pitch', '_yaw'])
line[0].set_dashes([2,2,10,2])
line[0].set_dash_capstyle('round')
line[1].set_dashes([5,2,5,2])
line[1].set_dash_capstyle('round')
line[2].set_dashes([2,2,2,2])
line[2].set_dash_capstyle('round')
ax2.set_ylabel('Spin (deg/s)', fontsize=tsS)
ax2.set_xlabel('Time (s)', fontsize=tsS)
ax2.axvspan(0, 0.1, color='lightgrey', alpha=0.5)
ax2.axvspan(0.2, 0.3, color='lightgrey', alpha=0.5)
ax2.axvspan(0.4, t[-1], color='lightgrey', alpha=0.5)
fig.legend(loc='outside right center')

plt.show()