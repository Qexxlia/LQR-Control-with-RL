import numpy as np

a = np.ones((3))

a = np.atleast_2d(a).T

b = np.zeros((6,2))

c = np.hstack(a,b[0:3])

print(a)
print(b)
print(c)