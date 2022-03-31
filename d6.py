import math
import numpy as np

from Satelite import DynamicSystem
from gps_plot import plot_satelites

if __name__ == '__main__':
    ds = DynamicSystem(n=8)
    max_cop, max_emf, _, new = ds.compute_EMF(np.array([0,0,6370,0.0001]))
    print(f"Maximum position error found: {max_cop} meters")
    print(f"condition number of the problem: {max_emf}")

    new[2] += DynamicSystem.earth_radius

    plot_satelites(ds, new)
