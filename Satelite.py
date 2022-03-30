import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from random import uniform
from Methods import Jacobi_row, e_mach



@dataclass
class Satelite:
    A: float
    B: float
    C: float 

    def get_pos(self): 
        return np.array([self.A, self.B, self.C])

@dataclass
class Receiver:
    A: float
    B: float
    C: float

    t: list[float]
    
class DynamicSatelite(Satelite):

    def __init__(self, phi: float, theta: float, rho: float =26570, d: float = 0.0001, z: float = 6370) -> 'DynamicSatelite':
        assert phi >= 0 and phi <= math.pi/2, "phi not in range"
        assert theta >= 0 and theta <= 2*math.pi, "theta not in range"

        A: float = rho*math.cos(phi)*math.cos(theta)
        B: float = rho*math.cos(phi)*math.sin(theta)
        C: float = rho*math.sin(phi) - z

        R = math.sqrt(A**2 + B**2 + C**2)
        t = d + R/SateliteSystem.speed_of_light #
        super().__init__(A,B,C) # 


   
class SateliteSystem(ABC):

    earth_radius: int = 6371 # km
    speed_of_light: float = 299792.458 # km/s

    def __init__(self,*satelites): #(Ai,Bi,Ci,ti, d)
        self.satelites: tuple[Satelite] = np.array(satelites)
      


    @abstractmethod
    def solve(self, unknown: np.ndarray) -> np.ndarray:
        """ 
            Solves the system of equations according to the given travel times
            and the current positions of the satelites.
        """
        
    def get_radii(self, unknowns: np.ndarray, dt: float) -> Callable[[Satelite], float]:
        return lambda satelite : math.sqrt(np.sum( (unknowns[:-1] - satelite.get_pos())**2 )) - SateliteSystem.speed_of_light*(dt - unknowns[-1])


class StaticSystem(SateliteSystem):
    
    def F(self, unknowns: np.ndarray, dt: float) -> np.ndarray:
        solution = list(map(self.get_radii(unknowns, dt), self.satelites))
        return np.array(solution)

    def DF(self,x):
        return np.hstack(
            (
                np.array([Jacobi_row(x[:-1], s.get_pos()) for s in self.satelites]),
                np.expand_dims([SateliteSystem.speed_of_light for _ in range(len(self.satelites))], axis=1)
            )
        )

    def solve(self, x0, dt) -> np.ndarray:
        xk = x0
        x_old = np.zeros_like(x0)
        F_ = lambda x: self.F(x0, dt) + self.DF(x0)@(x - x0)
        while (np.any((xk-x_old) > e_mach)):
            s = np.linalg.solve(self.DF(xk), -F_(xk))
            x_old = xk
            xk = xk + s
        return xk

class DynamicSystem(SateliteSystem):

    def __init__(self,phi_values,theta_values) -> 'DynamicSystem':
        assert len(phi_values) == len(theta_values)

        self.satelites = [DynamicSatelite(phi = phi, theta = theta) for (phi, theta) in zip(phi_values, theta_values)]


    def solve(self) -> tuple[float]:
        print("dynamic")

    # time/T fyrir réttan tíma
    def getPos(self, time: float) -> tuple[float]:

        phi, theta = self.phi*time+self.offset_phi, self.theta*time+self.offset_theta

        pos = (math.cos(phi)*math.cos(theta),
               math.cos(phi)*math.sin(theta),
               math.sin(theta))

        return (self.altitude*x for x in pos)



r = Receiver(1,2,3, [0.1, 0.2, 0.3])
print(r.t)
# d = DynamicSatelite(phi=0,theta=0)
# sys = StaticSystem(*(sat1, sat2, sat3,sat4))
# print(sys.solve(np.array([0,0,6370,0])))

# centers = np.array([(15600, 7540, 20140), (18760, 2750, 18610), (17610, 14630, 13480), (19170, 610, 18390)])
# satelites = [StaticSatelite(*x,t) for x,t in zip(centers, (0.07074, 0.07220, 0.07690, 0.07242))]

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
