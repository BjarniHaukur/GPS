import numpy as np
import math
from Satelite import SateliteConnection, SateliteSystem
from gps_plot import plot_satelites



if __name__ == '__main__':
    sat1 = SateliteConnection(15600,7540,20140,0.07074)
    sat2 = SateliteConnection(18760,2750,18610,0.07220)
    sat3 = SateliteConnection(17610,14630,13480,0.07690)
    sat4 = SateliteConnection(19170,610,18390,0.07242)
    
    sys = SateliteSystem([sat1, sat2, sat3,sat4])

    init_guess = np.array([0,0,6370,0])
    GN_pos = sys.solve_GN(init_guess)
    mN_pos = sys.solve_multivariate(init_guess)

    print(GN_pos)
    print(math.sqrt(sum([x**2 for x in GN_pos])))
    print(mN_pos)
    print(math.sqrt(sum([x**2 for x in mN_pos])))
    
    # plot_satelites(sys, r_pos, name="d1")
    d1_sol = np.array([-41.77271,-16.78919,6370.0596, 0.003201566])
    print("diff: ", np.sum(np.sqrt((d1_sol - GN_pos)**2) ))




