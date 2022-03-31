import numpy as np
import math
from Satelite import DynamicSystem
from gps_plot import plot_satelites
from test_fixture import TestGps

if __name__ == '__main__':
    
    num_iterations: int = 50
    t_err_min: float = 10**(-12)
    t_err_max: float = 10**(-8)

    receiver = (2, 1)
    test = TestGps(receiver)

    satelites = test.get_satelites(phi_diff=math.pi/4, theta_diff=math.pi/4, n=4)
    guess = np.array(test.get_initial_guess())

    ds = DynamicSystem(satelites)

    pe, emf = ds.compute_EMF(np.array(guess), t_err_min, t_err_max, num_iterations)
    print(f"With error rates ranging from {t_err_min} to {t_err_max} and {num_iterations} iterations we got:")
    print(f" a minimum position error of: {min(pe)*1000:.2f} meters,")
    print(f" an average position error of: {sum(pe)/len(pe)*1000:.2f} meters,")
    print(f" a maximum position error of: {max(pe)*1000:.2f} meters,")
    print(f" and the condition number of the problem is: {max(emf)}")

    # pos = ds.solve_GN(np.array(guess))
    # print(pos)
    # print(test.get_receiver_pos())
    # print("error: ", math.sqrt(sum([(x-y)**2 for x,y in zip(pos[:-1], test.get_receiver_pos())])))
    # plot_satelites(ds, pos)