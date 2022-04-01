import math
import numpy as np
from dataclasses import dataclass
from typing import Callable
from random import randint, uniform
from Methods import Jacobi_row, distance, e_mach


@dataclass
class SateliteConnection:
    """
    dataclass representing a connection to a satelite.
    
    """
    A: float
    B: float
    C: float 
    t: float

    def get_pos(self): 
        return np.array([self.A, self.B, self.C])


class SateliteSystem:
    """
    A system to calculate receiver position based on satelites.
    
    Class variables:
        earth_radius: Radius of earth in kilometers
        speed_of_light: Speed of light in kilmeters/second
    Attributes:
        satelites: a list of 'SateliteConnections'

    Methods:
        get_satelite: Returns a list of all satelites of the system.
        
    """
    earth_radius: int = 6370 # km
    speed_of_light: float = 299792.458 # km/s
    
    def __init__(self, sateliteConnections):
        self.satelites: list[SateliteConnection] = sateliteConnections

    def get_satelites(self):
        return self.satelites

    def _r(self, unknowns: np.ndarray) -> np.ndarray:
        solution = list(map(self._get_radii(unknowns), self.satelites))
        return np.array(solution)

    def _jacobi_matrix(self,x):
        """
        Input:
            x: unknowns, (x,y,z,d) Estimated position of the receiver
        Returns
        ------
        Jacobi matrix containing all partial derivates for each r_i
        """
        return np.hstack(
            (
                np.array([Jacobi_row(x[:-1], xy.get_pos()) for xy in self.satelites]),
                np.expand_dims(np.array([SateliteSystem.speed_of_light for _ in range(len(self.satelites))]), axis=1)
            )
        )

    def _get_radii(self, unknowns: np.ndarray) -> Callable[[SateliteConnection], float]:
        """
        Input:
            unknowns: (x,y,z,d), position estimate of the receiver
        Returns
        --------
        Callable function, corresponding to the r function of the Gauss-Newton method.
        """
        return lambda sateliteConnection : math.sqrt(np.sum( (unknowns[:-1] - sateliteConnection.get_pos())**2 )) - SateliteSystem.speed_of_light*(sateliteConnection.t - unknowns[-1])

    def solve_GN(self,position: np.ndarray) -> np.ndarray:
        """
        Input:
            position: (x,y,z,d) initial guess of the receivers position
        Returns
        -------
        (x,y,z,d) values of the receiver according to the Gauss-Newton method.
        """
        curr_pos = position
        old_pos = np.zeros_like(position)

        iteration = 0
        #GaussNewton method
        while (np.any((curr_pos-old_pos) > e_mach) and iteration < 1000):
            A = self._jacobi_matrix(curr_pos)
            
            v = np.linalg.solve(A.T@A,-A.T@self._r(curr_pos))
            old_pos = curr_pos
            curr_pos = curr_pos + v
            iteration += 1
        return curr_pos

    def solve_multivariate(self, position) -> np.ndarray:
        """ 
        Input:
            position: (x,y,z,d) initial guess of the receiver's position
        Returns
        -------
        (x,y,z,d) values of the receiver according to the multivariate newton's method.
        """

        #Multivariate newton's method
        curr_pos = position
        old_pos = np.zeros_like(position)
        F_ = lambda x: self.r(position) + self._jacobi_matrix(position)@(x - position)  
        is_square = len(self.satelites)==4  
        iteration = 0
        while (np.any((curr_pos-old_pos) > e_mach) and iteration < 1000):
            if is_square:
                s = np.linalg.solve(self._jacobi_matrix(curr_pos), -F_(curr_pos))
            else:
                inv_matrix = np.linalg.pinv(self._jacobi_matrix(curr_pos))
                s = inv_matrix@(-F_(curr_pos))
            old_pos = curr_pos
            curr_pos = curr_pos + s
            iteration += 1
        return curr_pos


class DynamicSystem(SateliteSystem):
    """
    Inherits 'SateliteSystem'

    """
    def compute_EMF(self, position: np.ndarray, t_err_min: float = 10**(-12), t_err_max: float = 10**(-8)) -> tuple[float, float]:
        """
        Input:
            position: Initial position guess (x,y,z,d)
            t_err_min: Minimum value for delta_t 
            t_err_max: Maximum value for delta_t
        Returns
        ------
        (Position Error, Error Magnification Factor)
        """
        old_pos = self.solve_GN(position)
        old_pos_times = np.array([satelite.t for satelite in self.satelites])
        diff_pos_times = np.zeros_like(old_pos_times)
        for i,satelite in enumerate(self.satelites):
            t_error = uniform(t_err_min, t_err_max)*(-1)**randint(0,1) #different values for ti
            satelite.t += t_error # we reinitialize the class after using this function
            diff_pos_times[i] = np.abs(t_error)

        new_pos = self.solve_GN(position)
        pos_change = np.abs(old_pos - new_pos)
        emf = np.amax(pos_change)/(SateliteSystem.speed_of_light*np.amax(diff_pos_times))
        position_error = distance(new_pos, old_pos)

        return position_error, emf


