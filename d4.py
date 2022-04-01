import numpy as np
import math
from Satelite import DynamicSystem, SateliteConnection
from gps_plot import plot_satelites
from test_fixture import TestGps, run_tests
from tabulate import tabulate



if __name__ == '__main__':
    receivers =  [(0,0), (-2,1), (2,1), (1.5, 0), (12,20)]
    

    df_lin = run_tests(receivers, phi_diff=math.pi/4, theta_diff=math.pi/4, iters=20, n_sat=4, random=False)

    print(df_lin)


    print(tabulate(df_lin, tablefmt="latex", floatfmt=".2f"))

    test = TestGps((2,1))
    guess = test.get_initial_guess()
    ds = DynamicSystem(test.get_random_satelites(math.pi/4, math.pi/4, n=4))
    pos = ds.solve_GN(guess)
    print(pos)
    print(test.get_receiver_pos())
    print("error: ", math.sqrt(sum([(x-y)**2 for x,y in zip(pos[:-1], test.get_receiver_pos())])))
    plot_satelites(ds, pos, name="d4")


