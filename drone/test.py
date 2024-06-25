import numpy as np
import matplotlib.pyplot as plt
import drone_dynamics as dd
import control

np.printoptions(precision=3, suppress=False, linewidth=140)

A,B = dd.precalcMatrices(1, 0.11, 0.11, 0.04)

print(A)
print("\n")
print(B)
print("\n")

Q = np.eye(12)
R = np.eye(4)

state = np.array([
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
], dtype=np.float32)


input()
[K,S,R] = control.lqr(A,B,Q,R)
print(K)

input()
fig, ax = plt.subplot(1,1)
ax[0].plot(sol.t, sol.y[0])

plt.show()

if 1 == 2:
  print(1)