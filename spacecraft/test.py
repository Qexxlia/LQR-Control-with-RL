import numpy as np
from scipy import integrate

pos = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 5, 6],
        [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6],
        [4, 4, 4, 4, 4, 1, 4, 2, 3, 4, 5, 6],
    ]
)

t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


res = np.sum(integrate.simpson(pos, x=t))

print(res)

