import numpy as np
import math
from Satelite import DynamicSystem
from gps_plot import plot_satelites

if __name__ == '__main__':
    
    num_iterations: int = 50
    t_err_min: float = 10**(-12)
    t_err_max: float = 10**(-8)

    ds = DynamicSystem(n=8)
    guess = np.array([6000,2000,1000,0.0001])
    pe, emf = ds.compute_EMF(np.array(guess), t_err_min, t_err_max, num_iterations)
    pos = ds.solve(np.array(guess))
    print(f"With error rates ranging from {t_err_min} to {t_err_max} and {num_iterations} iterations we got:")
    print(f" a minimum position error of: {min(pe)*1000:.2f} meters,")
    print(f" an average position error of: {sum(pe)/len(pe)*1000:.2f} meters,")
    print(f" a maximum position error of: {max(pe)*1000:.2f} meters,")
    print(f" and the condition number of the problem is: {max(emf)}")
    
    print(pos)
    plot_satelites(ds, pos)