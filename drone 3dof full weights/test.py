import numpy as np
import matplotlib.pyplot as plt
import drone_dynamics as dd

a = np.reshape(500 * np.ones(36), (6,6))
print(a)
b = a.T @ a
print(b)