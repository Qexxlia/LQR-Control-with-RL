import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import drone_dynamics as dd



desired_angle = 0.0872664625997165

# Define the initial state
initial_state = np.array([
    -desired_angle, # r
    -desired_angle, # p
    -desired_angle, # y
    0, # dr
    0, # dp
    0, # dy
], dtype=np.float32)

# print(initial_state)

u = None

if u is None:
    u = np.array([0, 0, 0, 0], dtype=np.float32)
    
if u is None:
    u = np.array([0, 0, 0, 0], dtype=np.float32)
