import math
import numpy as np

from Satelite import DynamicSystem

if __name__ == '__main__':
    ds = DynamicSystem()
    max_cop, max_emf = ds.compute_EMF(np.array([0,0,6370,0.0001]))
    print(f"Maximum position error found: {max_cop} meters")
    print(f"condition number of the problem: {max_emf}")

