import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from Methods import Jacobi_row,e_mach



@dataclass
class Satelite:
    A: float
    B: float
    C: float 
    t: float

    def get_pos(self): 
        return np.array([self.A, self.B, self.C])
    
class DynamicSatelite:

    def __init__(self, phi, theta, rho =26570 ) -> Satelite:
        assert phi >= 0 and phi <= math.PI/2, "theta not in range"
        assert theta >= 0 and theta <= math.PI/2, "PHI not in range"
        return Satelite(
            A = rho*math.cos(phi)*math.cos(theta),
            B = rho*math.cos(phi)*math.sin(theta),
            C = rho*math.sin(phi)
        )

   
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
        
    def get_radii(self, unknowns: np.ndarray) -> Callable[[Satelite], float]:
        return lambda satelite : math.sqrt(np.sum( (unknowns[:-1] - satelite.get_pos())**2 )) - SateliteSystem.speed_of_light*(satelite.t - unknowns[-1])


class StaticSystem(SateliteSystem):
    
    def F(self, unknowns: np.ndarray) -> np.ndarray:
        solution = list(map(self.get_radii(unknowns), self.satelites))
        return np.array(solution)
    def DF(self,x):
        return np.hstack(
            (
                np.array([Jacobi_row(x[:-1], xy.get_pos()) for xy in self.satelites]),
                np.array([SateliteSystem.speed_of_light for _ in range(len(self.satelites))]).reshape(4,1)
            )
        )
    def solve(self,x0) -> np.ndarray:
        xk = x0
        x_old = np.zeros_like(x0)
        F_ = lambda x: self.F(x0) + self.DF(x0)@(x - x0)
        while (np.any((xk-x_old) > e_mach)):
            s = np.linalg.solve(self.DF(xk), -F_(xk))
            x_old = xk
            xk = xk + s
        return xk

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
    def getPos(self, time: float) -> tuple[float]:

        phi, theta = self.phi*time+self.offset_phi, self.theta*time+self.offset_theta

        pos = (math.cos(phi)*math.cos(theta),
               math.cos(phi)*math.sin(theta),
               math.sin(theta))

        return (self.altitude*x for x in pos)

