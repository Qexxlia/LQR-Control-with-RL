import numpy as np

b = np.array([1,2,3], dtype=np.float32)

from SpacecraftEnv import SpacecraftEnv as spe

SpacecraftEnv = spe()
a = spe.map_range(spe, b, 0, 10, 10, 20)

print(a)