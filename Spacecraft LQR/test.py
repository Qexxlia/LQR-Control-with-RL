import numpy as np
from numpy import array
import math
import matplotlib.pyplot as plt
import SpacecraftDynamics as scd
import control

A = scd.calcAMatrix(6371 + 500, 3.986e5)

B = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1]
])

R = np.diag([0.01,0.01,0.01])
Q = np.diag([0.01,0.01,0.01,0.01,0.01,0.01])

print(A)
print(B)
print(R)
print(Q)

[K, S, E] = control.lqr(A, B, Q, R)

print("K: ", K)

a = K @ np.array([1,1,1,1,1,1])
b = K @ -np.array([1,1,1,1,1,1])
print(a)
print(b)

print(np.linalg.norm(a))