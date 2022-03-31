import numpy as np
import math
from Satelite import SateliteConnection, SateliteSystem
from gps_plot import plot_satelites



if __name__ == '__main__':
    sat1 = SateliteConnection(15600,7540,20140,0.07074)
    sat2 = SateliteConnection(18760,2750,18610,0.07220)
    sat3 = SateliteConnection(17610,14630,13480,0.07690)
    sat4 = SateliteConnection(19170,610,18390,0.07242)
    
    sys = SateliteSystem(*(sat1, sat2, sat3,sat4))
    r_pos = sys.solve(np.array([0,0,6370,0]))

    print(math.sqrt(sum([x**2 for x in r_pos])))
    print(r_pos)
    plot_satelites(sys, r_pos, name="d1")




