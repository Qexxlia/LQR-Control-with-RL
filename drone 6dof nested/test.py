import numpy as np

a = np.array([1,1,1,1])
b = np.array([1,1])

print(a-np.pad(b, (0, len(a)-len(b)), 'constant'))