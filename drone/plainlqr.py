import drone_dynamics as dd
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl


state = np.array([
    1, # x
    1, # y
    0, # z
    0, # roll
    0, # pitch
    0, # yaw
    0, # x_dot
    0, # y_dot
    0, # z_dot
    0, # roll_dot
    0, # pitch_dot
    0, # yaw_dot
], dtype=np.float32)

time_range = (0, 10)

q_weights = np.array([100,400,500,1.123,1.1234,1.1243,0.413,0.143,193,0.2134,134,1324], dtype=np.float32)
r_weights = np.array([103,484,601,484], dtype=np.float32)

A, B = dd.precalcMatrices(1, 0.11, 0.11, 0.04)
Q = np.diag(q_weights)
R = np.diag(r_weights)

print(A)
print(B)
print(Q)
print(R)

u_max = 1

ctrl.care(A, B, Q, R, method='slycot')
print("HERE")

try:
    [K,S,P] = ctrl.lqr(A, B, Q, R, method='slycot')
except Exception as e:
    print("exception")
    print(e)
print(K)


# sol = dd.simulate(state, time_range, q_weights, r_weights, A, B, u_max)

# plt.plot(sol.t)
# plt.show()

input("exit")