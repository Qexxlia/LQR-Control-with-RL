import warnings

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem, Problem, StarmapParallelization
from pymoo.optimize import minimize
from scipy import integrate

import spacecraft_dynamics as scd

np.set_printoptions(precision=5, linewidth=10000)
warnings.filterwarnings("ignore")


class SimulateSpacecraft(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=9, n_obj=1, xl=1, xu=1e6, **kwargs)
        self.state = np.array(
            [
                0.5,  # x
                -0.5,  # y
                0,  # z
                1e-3,  # x_dot
                -1e-3,  # y_dot
                0,  # z_dot
                30,  # mass
            ],
        )

        self.A, self.B = scd.precalcMatrices(6371 + 500, 3.986e5)

    def _evaluate(self, x, out, *args, **kwargs):
        sol = scd.simulate(
            self.state, (0, 1000), x[0:6], x[6:9], self.A, self.B, 1e-5, 15
        )

        out["F"] = (
            integrate.simpson(
                (sol.y[0, :] ** 2 + sol.y[1, :] ** 2 + sol.y[2, :] ** 2), x=sol.t
            )
            + sol.t[-1]
        )


pop_size = 1000

problem = SimulateSpacecraft()
print("pop_size=", pop_size)


alg = PSO(pop_size=pop_size)

res = minimize(problem, alg, seed=1, verbose=True)

print("X:", res.X)
print("F:", res.F)
