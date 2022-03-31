import numpy as np
from Satelite import SateliteConnection, SateliteSystem
from gps_plot import plot_satelites



if __name__ == '__main__':
    sat1 = SateliteConnection(15600,7540,20140,0.07074)
    sat2 = SateliteConnection(18760,2750,18610,0.07220)
    sat3 = SateliteConnection(17610,14630,13480,0.07690)
    sat4 = SateliteConnection(19170,610,18390,0.07242)
    
    sys = SateliteSystem(*(sat1, sat2, sat3,sat4))
    r_pos = sys.solve2(np.array([0,0,6370,0]))
    print(f"x= {r_pos[0]}, y= {r_pos[1]}, z= {r_pos[2]}, d= {r_pos[3]}")
    plot_satelites(sys, r_pos, name="d1")
    d1_sol = np.array([-41.77271,-16.78919,6370.0596, 0.003201566])
    print("diff: ", np.sum(np.sqrt((d1_sol - r_pos)**2) ))




