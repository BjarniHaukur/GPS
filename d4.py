import math
import numpy as np

from Satelite import DynamicSystem
if __name__ == '__main__':
    ds = DynamicSystem()
    ds.compute_EMF(np.array([0,0,6370,0.0001]))

