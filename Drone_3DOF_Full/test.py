import numpy as np
import matplotlib.pyplot as plt
import drone_dynamics as dd


def log_map_range(val, in_min, in_max, out_min, out_max): 
    return np.copysign(out_min * (out_max/out_min)**((abs(val) - in_min)/(in_max - in_min)), val) # LOG

a = log_map_range(-1, 0, 1, 0.1, 1000)
b = log_map_range(1, 0, 1, 0.1, 1000)

print(a)
print(b)