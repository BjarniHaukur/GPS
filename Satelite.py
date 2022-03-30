import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

@dataclass
class Satelite:
    A: float
    B: float
    C: float 
    t: float

    def get_pos(self): 
        return np.array([self.A, self.B, self.C])
    
   
class SateliteSystem(ABC):

    earth_radius: int = 6371 # km
    speed_of_light: float = 299792.458 # km/s

    def __init__(self, *satelites): #(Ai,Bi,Ci,ti, d)
        self.satelites = np.array(satelites)

    @abstractmethod
    def solve(self, unknown: np.ndarray) -> np.ndarray:
        """ 
            Solves the system of equations according to the given travel times
            and the current positions of the satelites.
        """
        
    def get_radii(self, unknowns: np.ndarray) -> Callable[[Satelite], float]:
        return lambda satelite : math.sqrt(np.sum((unknowns[:-1] - satelite.get_pos())**2)) - SateliteSystem.speed_of_light*(satelite.t - unknowns[-1])


class StaticSystem(SateliteSystem):

    def solve(self, unknowns: np.ndarray) -> np.ndarray:
        solution = list(map(self.get_radii(unknowns), self.satelites))
        return np.array(solution)

class DynamicSystem(SateliteSystem):

    def __init__(self, phi: float, theta: float,
                 offset_phi: float = 0, offset_theta: float = 0, altitude: int = 20200):

        self.radius = altitude + SateliteSystem.earth_radius
        self.phi = phi
        self.theta = theta
        self.offset_phi = offset_phi
        self.offset_theta = offset_theta
    
    def solve(self) -> tuple[float]:
        print("dynamic")

    # time/T fyrir réttan tíma
    def get_pos(self, time: float) -> tuple[float]:

        phi, theta = self.phi*time+self.offset_phi, self.theta*time+self.offset_theta

        pos = (math.cos(phi)*math.cos(theta),
               math.cos(phi)*math.sin(theta),
               math.sin(theta))

        return (self.altitude*x for x in pos)



centers = np.array([(15600, 7540, 20140), (18760, 2750, 18610), (17610, 14630, 13480), (19170, 610, 18390)])
satelites = [Satelite(*x,t) for x,t in zip(centers, (0.07074, 0.07220, 0.07690, 0.07242))]

print(satelites[0].t)

sys = StaticSystem(*satelites)
print(sys.solve(np.array([0, 0, 6370, 0])))


#print(stat.solve(0.07074, 0.07220, 0.07690, 0.07242))
# dyn = DynamicSatelite(1,2)
# dyn.solve(1,2,3)
# print(Satelite.earth_radius)

# def main():
#     x0 = np.array((0, 0, 0))
#     centers1 = np.array([(0, 1, 1), (1,1, 1), (0,-1, 100)])
#     radii1 = np.array([1, 1, 1])
#     # centers2 = np.array([(-1,0), (1,1), (1,-1)])
#     # radii2 = np.array([1, 1, 1])
    

#     x1 = GaussNewton(x0, centers1, radii1)
#     print(f"Least square distance for example 1 is at: {x1}")
#     # x2 = GaussNewton(x0, centers2, radii2)
#     # print(f"Least square distance for example 2 is at: {x2}")

# if __name__ == "__main__":
#     main()