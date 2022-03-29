import math
import numpy as np
from abc import ABC, abstractmethod
from Methods import GaussNewton, distance, e_mach, Jacobi_row

Point = list[float, 3] # Er í raun np.array
Vector = list[Point] # -||-

class Satelite(ABC):

    earth_radius: int = 6371 # km
    speed_of_light: float = 99792.458 # km/s

    def __init__(self, A: Point, B: Point, C: Point, t: Point): #(Ai,Bi,Ci,ti, d)
        self.A: Point = A
        self.B: Point = B
        self.C: Point = C
        self.t: Point = t

        self.vars = np.array([self.A,self.B,self.C,self.t])

    def get_radii(self, unknowns: np.ndarray) -> np.float64: #unknowns = (x,y,z,d)

        return math.sqrt(np.sum(unknowns[:-1] - self.vars[:-1])**2) - self.speed_of_light*(self.t - unknowns[-1]) #reikna r_i
   
    @abstractmethod
    def solve(self, At: float, Bt: float, Ct: float, Dt: float) -> tuple[float]:
        """ 
            Solves the system of equations according to the given travel times
            and the current positions of the satelites.
        """
        ...

class StaticSatelite(Satelite):

    def solve(self, At: float, Bt: float, Ct: float, Dt: float) -> tuple[float]:
        if self.time_dilation is None:
            self.time_dilation = Dt - distance(self.D, [0]*3)/self.speed_of_light

        return GaussNewton(np.array([0, 0, 6370]), np.c_[self.A, self.B, self.C],
                          (np.array([At, Bt, Ct])-self.time_dilation)*self.speed_of_light)
        


class DynamicSatelite(Satelite):

    def __init__(self, phi: float, theta: float,
                 offset_phi: float = 0, offset_theta: float = 0, altitude: int = 20200) -> "Satelite":

        self.radius = altitude + Satelite.earth_radius
        self.phi = phi
        self.theta = theta
        self.offset_phi = offset_phi
        self.offset_theta = offset_theta
    
    def solve(self, At: float, Bt: float, Ct: float) -> tuple[float]:
        print("dynamic")

    # time/T fyrir réttan tíma
    def getPos(self, time: float) -> tuple[float]:

        phi, theta = self.phi*time+self.offset_phi, self.theta*time+self.offset_theta

        pos = (math.cos(phi)*math.cos(theta),
               math.cos(phi)*math.sin(theta),
               math.sin(theta))

        return (self.altitude*x for x in pos)



centers = np.array([(15600, 7540, 20140), (18760, 2750, 18610), (17610, 14630, 13480), (19170, 610, 18390)])
satelites = [StaticSatelite(*x,t) for x,t in zip(centers, (0.07074, 0.07220, 0.07690, 0.07242))]

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