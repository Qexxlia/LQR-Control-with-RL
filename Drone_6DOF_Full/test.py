import numpy as np
import scipy

import drone_dynamics as dd

np.set_printoptions(linewidth=100000)

A, B = dd.precalcMatrices()
Q = np.diag(np.ones((12)))
R = np.array(
    [
        [
            1.48313816e-12,
            1.39138130e-15,
            -4.12478598e-15,
            4.22749155e-15,
        ],
        [
            1.39138130e-15,
            2.45861365e-10,
            -2.35936267e-13,
            2.16014114e-13,
        ],
        [
            -4.12478598e-15,
            -2.35936267e-13,
            3.58120246e-12,
            9.47634421e-15,
        ],
        [
            4.22749155e-15,
            2.16014114e-13,
            9.47634421e-15,
            6.38716971e-11,
        ],
    ]
)

m, n = B.shape

H = np.empty((2 * m + n, 2 * m + n))
H[:m, :m] = A
H[:m, m : 2 * m] = 0.0
H[:m, 2 * m :] = B
H[m : 2 * m, :m] = -Q
H[m : 2 * m, m : 2 * m] = -A.conj().T
H[m : 2 * m, 2 * m :] = 0.0
H[2 * m :, :m] = 0.0
H[2 * m :, m : 2 * m] = B.conj().T
H[2 * m :, 2 * m :] = R

J = scipy.linalg.block_diag(np.eye(2 * m), np.zeros_like(R))

print("H:\n", H)
print("J:\n", J)

print(scipy.linalg.det(H))
print(scipy.linalg.det(J))

print(scipy.linalg.solve_continuous_are(A, B, Q, R))
