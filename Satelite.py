import math
import numpy as np

from Methods import GaussNewton



class StaticSatelite:

    earth_radius: int = 6371

    def __init__(self, A: float, B: float, C: float):
        self.A = A
        self.B = B
        self.C = C

    def solve(self, At: float, Bt: float, Ct: float) -> tuple[float]:
        print("solved")


class Satelite(StaticSatelite):

    def __init__(self, phi: float, theta: float,
                 offset_phi: float = 0, offset_theta: float = 0, altitude: int = 20200) -> "Satelite":

        self.radius = altitude + Satelite.earth_radius
        self.phi = phi
        self.theta = theta
        self.offset_phi = offset_phi
        self.offset_theta = offset_theta
    
    # t/T fyrir réttan tíma
    def getPos(self, time: float) -> tuple[float]:

        phi, theta = self.phi*time+self.offset_phi, self.theta*time+self.offset_theta

        pos = (math.cos(phi)*math.cos(theta),
               math.cos(phi)*math.sin(theta),
               math.sin(theta))

        return (self.altitude*x for x in pos)


# stat = StaticSatelite(1,2,3)
# stat.solve(1,2,3)
# sat = Satelite(1,2)
# sat.solve(1,2,3)
# print(Satelite.earth_radius)

def main():
    x0 = np.array((0, 0, 0))
    centers1 = np.array([(0, 1, 1), (1,1, 1), (0,-1, 100)])
    radii1 = np.array([1, 1, 1])
    # centers2 = np.array([(-1,0), (1,1), (1,-1)])
    # radii2 = np.array([1, 1, 1])
    

    x1 = GaussNewton(x0, centers1, radii1)
    print(f"Least square distance for example 1 is at: {x1}")
    # x2 = GaussNewton(x0, centers2, radii2)
    # print(f"Least square distance for example 2 is at: {x2}")

if __name__ == "__main__":
    main()