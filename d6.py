import numpy as np
import math
from Satelite import DynamicSystem, SateliteConnection
from gps_plot import plot_satelites
from test_fixture import TestGps, run_tests, SateliteGenerator
from functools import partial


if __name__ == '__main__':
    receivers =  [(0,0), (-2,1), (2,1), (1.5, 0), (12,20)]
    

    df_lin = run_tests(receivers, phi_diff=math.pi/4, theta_diff=math.pi/4, iters=20, n_sat=8, random=False)
    df_rand = run_tests(receivers, phi_diff=math.pi/4, theta_diff=math.pi/4, iters=20, n_sat=8, random=True)
    print(df_lin)
    print(df_rand)

    test = TestGps((2,1.00004))
    guess = test.get_initial_guess()
    ds = DynamicSystem(test.get_random_satelites(math.pi/4, math.pi/4, n=8))
    pos = ds.solve_GN(guess)
    print(pos)
    print(test.get_receiver_pos())
    print("error: ", math.sqrt(sum([(x-y)**2 for x,y in zip(pos[:-1], test.get_receiver_pos())])))
    plot_satelites(ds, pos)
    # receiver = (2, 1)
    # test = TestGps(receiver)
    
    # sat_gen_linspace: SateliteGenerator = partial(test.get_linspace_satelites, phi_diff=math.pi/4, theta_diff=0.75*math.pi/2, n=8)
    # sat_gen_random: SateliteGenerator = partial(test.get_random_satelites, phi_diff=math.pi/4, theta_diff=0.75*math.pi/2, n=8)  
    # df_lin = run_tests(test, sat_gen_linspace, n_in=50, n_out=5)
    # df_rand = run_tests(test, sat_gen_random, n_in=50, n_out=5)
    # print(df_lin)
    # print(df_rand)

    # # ds = DynamicSystem(sat_gen())
    # # pos = ds.solve_GN(test.get_initial_guess())
    # # print(pos)
    # # print(test.get_receiver_pos())
    # # print("error: ", math.sqrt(sum([(x-y)**2 for x,y in zip(pos[:-1], test.get_receiver_pos())])))
    # # plot_satelites(ds, pos)