import warnings

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem, Problem, StarmapParallelization
from pymoo.optimize import minimize

import drone_dynamics as dd

np.set_printoptions(precision=5, linewidth=10000)
warnings.filterwarnings("ignore")


class SimulateDroneDiagonal(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=16, n_obj=1, xl=1, xu=5500, **kwargs)
        self.state = np.array(
            [
                10,
                7,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )

        self.A, self.B = dd.precalcMatrices()

    def _evaluate(self, x, out, *args, **kwargs):
        Q = np.diag(x[0:12])
        R = np.diag(x[12:16])

        sol, u = dd.simulate(self.state, (0, 150), Q, R, self.A, self.B)

        if len(sol.t_events[1]) != 0:
            out["F"] = 1e8 * (1 + np.sum(abs(sol.y[:, -1])))
            return

        if sol.t[-1] < 150:
            out["F"] = sol.t[-1]
        else:
            out["F"] = sol.t[-1] + 1e8 * np.sum(abs(sol.y[:, -1]))


class SimulateDroneFull(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=88, n_obj=1, xl=-500, xu=500, **kwargs)
        self.state = np.array(
            [
                10,
                7,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )

        self.A, self.B = dd.precalcMatrices()

        # Action masks
        self.q_off_diagonal_mask = np.ones((12, 12), dtype=bool)
        self.q_off_diagonal_mask[np.triu_indices(12)] = False
        self.q_diagonal_mask = np.eye(12, dtype=bool)

        self.r_off_diagonal_mask = np.ones((4, 4), dtype=bool)
        self.r_off_diagonal_mask[np.triu_indices(4)] = False
        self.r_diagonal_mask = np.eye(4, dtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):

        Q, R = self.map_action(x)
        if Q is None:
            out["F"] = 1e10
            return

        sol, u = dd.simulate(self.state, (0, 150), Q, R, self.A, self.B)

        if len(sol.t_events[1]) != 0:
            out["F"] = 1e8 * (1 + np.sum(abs(sol.y[:, -1])))
            return

        if sol.t[-1] < 150:
            out["F"] = sol.t[-1]
        else:
            out["F"] = sol.t[-1] + 1e8 * np.sum(abs(sol.y[:, -1]))

    def map_action(self, x):
        q_off_diagonal_weights = x[0:66]

        q_diagonal_weights = x[66:78]

        r_off_diagonal_weights = x[78:84]

        r_diagonal_weights = x[84:88]

        q = np.zeros((12, 12))
        q[self.q_off_diagonal_mask] = q_off_diagonal_weights / 1000
        q[self.q_diagonal_mask] = q_diagonal_weights

        r = np.zeros((4, 4))
        r[self.r_off_diagonal_mask] = r_off_diagonal_weights / 1000
        r[self.r_diagonal_mask] = r_diagonal_weights

        Q = q @ q.T
        R = r @ r.T
        min_svd = np.linalg.svd(R, compute_uv=False)[-1]
        if min_svd == 0 or min_svd < np.spacing(1.0) * np.linalg.norm(R, 1):
            return None, None

        return Q, R


pop_size = 100

# pool = ThreadPool(pop_size)
# runner = StarmapParallelization(pool.starmap)
# problem = SimulateDroneFull(elementwise_runner=runner)

# problem = SimulateDroneFull()
# Q[Q == 0] = 1
# R[R == 0] = 1
# X = np.random.random((pop_size, problem.n_var))
# # X[0, :] = np.hstack((np.ndarray.flatten(Q), np.ndarray.flatten(R)))
# X[0, :] = S
# print("method=", "Full")


problem = SimulateDroneDiagonal()
# X = np.random.random((pop_size, problem.n_var))
# X[0, :] = np.hstack((np.diagonal(Q), np.diagonal(R)))
print("method=", "Diagonal")


print("pop_size=", pop_size)


alg = PSO(pop_size=pop_size)
# alg = G3PCX(pop_size=pop_size)
# alg = ISRES()

res = minimize(problem, alg, seed=1, verbose=True)

print("X:", res.X)
print("F:", res.F)
