import numpy as np
import math
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(1000):
    r = math.sqrt(2)
    theta = np.random.uniform(0, np.pi/2)
    phi = np.random.uniform(0, np.pi/2)

    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    ax.scatter(x,y,z)

plt.show()