import numpy as np

initialState = np.array([
    0.5,    # x
    -0.5,    # y
    0.5,    # z
    1e-3,   # x_dot
    -1e-3,   # y_dot
    1e-3,   # z_dot
    30   # mass 
], dtype=np.float32)

r = np.append(np.append(np.random.normal(0, 5e-2, 3), np.random.normal(0, 5e-5, 3)),0)
              

state = initialState + r

print(state)
print(state/initialState)
