from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
import numpy as np
import matplotlib.pyplot as plt
import drone_dynamics as dd

class DronePSO():
    def __init__(self, args, verbose):
        self.t_step_limits = args.get('t_step_limits')
        self.variance = args.get('variance')
        self.variance_type = args.get('variance_type')
        self.max_duration = args.get('max_duration')
        self.map_limits = args.get('map_limits')
        self.u_max = args.get('u_max')
        self.simulation_type = args.get('simulation_type')
        self.t_step = args.get('t_step')
        self.seed = args.get('seed')
        self.absolute_norm = args.get('absolute_norm')
        self.verbose = verbose

        self.initial_time = 0

        # Define the initial state
        self.initial_state = np.array([
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
        
        # Calculate A matrix for the dynamics
        (self.A, self.B) = dd.precalcMatrices(1, 0.11, 0.11, 0.04) # From F. Ahmed et al

        self.bounds = (np.ones(16), np.ones(16)*1000)
        
    def optimize(self):
        # Initialize the optimizer
        optimizer = GlobalBestPSO(n_particles=32, dimensions=16, options={'c1': 0.5, 'c2': 0.3, 'w':0.9}, bounds=self.bounds)
        
        # Perform the optimization
        cost, pos = optimizer.optimize(self.cost_function, iters=10000)
        
        plot_cost_history(cost_history=optimizer.cost_history)
        plt.show()
        
        return cost, pos
        

    def cost_function(self, x):
        time_range = (self.initial_time, self.initial_time + self.max_duration)
        cost = np.zeros(x.shape[0])
        
        for i in range(0, x.shape[0]):
            q_weights = x[i, 0:12]
            r_weights = x[i, 12:16]
            print(i)
            print(q_weights)
            print(r_weights)
            print("\n")
        
            sol = dd.simulate(self.initial_state, time_range, q_weights, r_weights, self.A, self.B, self.u_max)
        
            cost[i] = sol.t[-1]
            if(sol.t[-1] > self.max_duration - 1):
                cost[i] += -100*np.linalg.norm(sol.y[:,-1])
        
        return cost