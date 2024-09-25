import numpy as np
import csv
import matplotlib.pyplot as plt

diag = np.genfromtxt('diag.csv', delimiter=',')
full = np.genfromtxt('full.csv', delimiter=',')
    
plt.plot(diag[:,1], diag[:,2], label='Diagonal')
plt.plot(full[:,1], full[:,2], label='Diagonal')

plt.show()