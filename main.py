import numpy as np
from Methods import D1
from Satelite import StaticSatelite


if __name__ == '__main__':
    centers = np.array([(15600, 7540, 20140), (18760, 2750, 18610), (17610, 14630, 13480), (19170, 610, 18390)])
    satelites = (StaticSatelite(*x,t) for x,t in zip(centers, (0.07074, 0.07220, 0.07690, 0.07242)))
    d1 = D1(*satelites)
    print(d1.compute_receiver_position(np.array([0,0,6370,0])))