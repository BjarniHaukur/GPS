import numpy as np
from Satelite import Satelite, StaticSystem


if __name__ == '__main__':
    sat1 = Satelite(15600,7540,20140,0.07074)
    sat2 = Satelite(18760,2750,18610,0.07220)
    sat3 = Satelite(17610,14630,13480,0.07690)
    sat4 = Satelite(19170,610,18390,0.07242)
    
    sys = StaticSystem(*(sat1, sat2, sat3,sat4))
    print(sys.solve(np.array([0,0,6370,0])))