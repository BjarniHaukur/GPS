import numpy as np
from Satelite import SateliteConnection, StaticSystem


if __name__ == '__main__':
    sat1 = SateliteConnection(15600,7540,20140,0.07074)
    sat2 = SateliteConnection(18760,2750,18610,0.07220)
    sat3 = SateliteConnection(17610,14630,13480,0.07690)
    sat4 = SateliteConnection(19170,610,18390,0.07242)
    
    sys = StaticSystem(*(sat1, sat2, sat3,sat4))
    print(sys.solve(np.array([0,0,6370,0])))