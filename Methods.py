import math
import numpy as np

from Satelite import Satelite


class D1:
    Point = list[float, 3] # Er í raun np.array
    Vector = list[Point] # -||-
    e_mach = 2**-52 

    def __init__(self, *argv: tuple[Satelite]):
        self.satelites: tuple[Satelite] = argv
        
    def distance(p: Point, p0: Point) -> float:
        return math.sqrt(sum([(x - x0)**2 for (x, x0) in zip(p, p0)]))  

    def Jacobi_row(p: Point, p0: Point) -> Point:
        dist = D1.distance(p, p0)
        return (p-p0)/dist  

    def compute_receiver_position(self,x0: Point) -> Point:
        ##eigum að nota multivariate newton aðferðina í kafl 2.7
        

        x_old = np.zeros_like(x0)
        DF = lambda x: np.hstack(
            (
                np.array([D1.Jacobi_row(x[:-1], xy.vars[:-1]) for xy in self.satelites]),
                np.array([Satelite.speed_of_light for _ in range(len(self.satelites))]).reshape(4,1)
            )
        ) 

    
        F0 = np.array([satelite.get_radii(x0) for satelite in self.satelites])
        F = lambda x : F0 + DF(x0)@(x-x0)
        xk = x0
        while (np.any((xk-x_old) > D1.e_mach)):
            print(DF(xk))
            s = np.linalg.solve(DF(xk), -F(xk))
            x_old = xk
            xk = xk + s

        return xk

