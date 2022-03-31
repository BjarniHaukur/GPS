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

    d1_sol = np.array([-41.77271,-16.78919,6370.0596, 0.003201566])

    print("Coordinates from Gauss Newton:\n", GN_pos)
    print(f"length of vector: {math.sqrt(sum([x**2 for x in GN_pos]))}")
    print(f"error: {np.sum(np.sqrt((d1_sol - GN_pos)**2))*1000:.3f} meters")

    print()

    print("Coordinates from multivariate Newton's method:\n", mN_pos)
    print(f"length of vector: {math.sqrt(sum([x**2 for x in mN_pos]))}")
    print(f"error: {np.sum(np.sqrt((d1_sol - mN_pos)**2))*1000:.3f} meters")
    
    plot_satelites(sys, GN_pos, name="d1")
    




