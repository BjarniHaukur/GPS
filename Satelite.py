import math
import numpy as np
from dataclasses import dataclass
from typing import Callable
from random import randint, uniform
from Methods import Jacobi_row, distance, e_mach
from matplotlib import pyplot as plt


@dataclass
class SateliteConnection:
    A: float
    B: float
    C: float 
    t: float

    def get_pos(self): 
        return np.array([self.A, self.B, self.C])
    
class DynamicSateliteConnection(SateliteConnection):

    def __init__(self, phi: float, theta: float, rho: float =26570, d: float = 0.0001, z: float = 6370) -> 'DynamicSateliteConnection':
        assert phi >= 0 and phi <= math.pi/2, "phi not in range"
        assert theta >= 0 and theta <= 2*math.pi, "theta not in range"

        A: float = rho*math.cos(phi)*math.cos(theta)
        B: float = rho*math.cos(phi)*math.sin(theta)
        C: float = rho*math.sin(phi) - z

        R = math.sqrt(A**2 + B**2 + C**2)
        t = d + R/SateliteSystem.speed_of_light
        super().__init__(A,B,C,t)


   
class SateliteSystem:

    earth_radius: int = 6370 # km
    speed_of_light: float = 299792.458 # km/s
    
    def __init__(self, *sateliteConnections):
        self.satelites: list[SateliteConnection] = sateliteConnections

    def get_satelites(self):
        return self.satelites

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

    def get_radii(self, unknowns: np.ndarray) -> Callable[[SateliteConnection], float]:
        return lambda sateliteConnection : math.sqrt(np.sum( (unknowns[:-1] - sateliteConnection.get_pos())**2 )) - SateliteSystem.speed_of_light*(sateliteConnection.t - unknowns[-1])
    
    def solve(self,position) -> np.ndarray:
        """ 
            Solves the system of equations according to the given travel times
            and the current positions of the sateliteConnections.
        """
        curr_pos = position
        old_pos = np.zeros_like(position)
        F_ = lambda x: self.F(position) + self.DF(position)@(x - position)
        iteration = 0
        while (np.any((curr_pos-old_pos) > e_mach) and iteration < 1000):
            s = np.linalg.solve(self.DF(curr_pos), -F_(curr_pos))
            old_pos = curr_pos
            curr_pos = curr_pos + s
            iteration += 1
        return curr_pos

# class StaticSystem(SateliteSystem):
#     def __init__(self,*sateliteConnections): #(Ai,Bi,Ci,ti, d)
#         self.satelites: tuple[SateliteConnection] = sateliteConnections

    
    


class DynamicSystem(SateliteSystem):

    def __init__(self, theta_min = 0, theta_max = 2*math.pi, phi_min = 0, phi_max = math.pi/2, n = 4) -> 'DynamicSystem':

        self.args = (theta_min,theta_max,phi_min,phi_max,n) #used for reinitialization

        theta_values = np.linspace(theta_min, theta_max, num=n)
        phi_values = np.linspace(phi_min, phi_max, num=n)
        super().__init__(*(DynamicSateliteConnection(phi = phi, theta = theta) for (phi, theta) in zip(phi_values, theta_values)))

    def compute_EMF(self, position: np.ndarray, t_err_min: float = 10**(-12), t_err_max: float = 10**(-8), num_iterations = 10) -> float: #t_error á mögulega að vera mismunandi gildi.. ?? 
        max_cop = float('-inf')
        max_emf = float('-inf')
        for _ in range(num_iterations):
            old_pos = self.solve(position)
            old_pos_times = np.array([satelite.t for satelite in self.satelites])

            diff_pos_times = np.zeros_like(old_pos_times)
            for i,satelite in enumerate(self.satelites):
                t_error = uniform(t_err_min, t_err_max)*(-1)**randint(0,1) #different values for ti
                satelite.t += t_error
                diff_pos_times[i] = np.abs(t_error)

            new_pos = self.solve(position)

            pos_change = np.abs(old_pos - new_pos)
            emf = np.amax(pos_change)/(SateliteSystem.speed_of_light*np.amax(diff_pos_times))
            cop = distance(new_pos, old_pos)
            max_cop = max(max_cop,cop)
            max_emf = max(max_emf, emf)
            self.__init__(*self.args)
        return max_cop*1000, max_emf




