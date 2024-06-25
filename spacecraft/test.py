import numpy as np
import matplotlib.pyplot as plt

num_time_steps = 100
step_gap = 20

initial = 3
final = 0.3

a = np.linspace(0, num_time_steps, (int)(num_time_steps/step_gap) + 1)
print(a)

b = np.linspace(initial, final, (int)(num_time_steps/step_gap))
print(b)

plt.stairs(b, a)

plt.show()