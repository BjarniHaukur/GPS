import numpy as np
import math
from Satelite import DynamicSystem, SateliteConnection
from gps_plot import plot_satelites
from test_fixture import TestGps, run_tests, SateliteGenerator




if __name__ == '__main__':
    receivers =  [(0,0), (-2,1), (2,1), (1.5, 0), (12,20)]
    

    df_lin = run_tests(receivers, phi_diff=math.pi/80, theta_diff=math.pi/20, iters=20, n_sat=4, random=False)

    print(df_lin)

    test = TestGps((2,1.00004))
    guess = test.get_initial_guess()
    ds = DynamicSystem(test.get_random_satelites(math.pi/80, math.pi/80, n=4))
    pos = ds.solve_GN(guess)
    print(pos)
    print(test.get_receiver_pos())
    print("error: ", math.sqrt(sum([(x-y)**2 for x,y in zip(pos[:-1], test.get_receiver_pos())])))
    plot_satelites(ds, pos)

    # receiver = (2, 2)
    # test = TestGps(receiver)
    
    # sat_gen_linspace: SateliteGenerator = partial(test.get_linspace_satelites, phi_diff=math.pi/80, theta_diff=math.pi/20, n=4)
    # sat_gen_random: SateliteGenerator = partial(test.get_random_satelites, phi_diff=math.pi/80, theta_diff=math.pi/20, n=4)  
    # df_lin = run_tests(test, sat_gen_linspace, n_in=50, n_out=5)
    # df_rand = run_tests(test, sat_gen_random, n_in=50, n_out=5)
    # print(df_lin)
    # print(df_rand)

    # ds = DynamicSystem(sat_gen_random())
    # print(ds.satelites[0].t)
    # pos = ds.solve_GN(test.get_initial_guess())
    # print(pos)
    # print(test.get_receiver_pos())
    # print("error: ", math.sqrt(sum([(x-y)**2 for x,y in zip(pos[:-1], test.get_receiver_pos())])))
    # plot_satelites(ds, pos)