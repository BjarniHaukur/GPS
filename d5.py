import numpy as np

from Satelite import DynamicSystem
from gps_plot import plot_satelites

if __name__ == '__main__':
    
    num_iterations: int = 100
    t_err_min: float = 10**(-12)
    t_err_max: float = 10**(-8)

    ds = DynamicSystem()
    pe, emf = ds.compute_EMF(np.array([0,0,6370,0.0001]), t_err_min, t_err_max, num_iterations)
    
    print(f"With error rates ranging from {t_err_min} to {t_err_max} and {num_iterations} iterations we got:")
    print(f" a minimum position error of: {min(pe)*1000:.2f} meters,")
    print(f" an average position error of: {sum(pe)/len(pe)*1000:.2f} meters,")
    print(f" a maximum position error of: {max(pe)*1000:.2f} meters,")
    print(f" and the condition number of the problem is: {max(emf)}")

 