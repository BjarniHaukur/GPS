import math

class Satelite:

    earth_radius: int = 6371

    def __init__(self, phi: float, theta: float,
                 offset_phi: float = 0, offset_theta: float = 0, altitude: int = 20200) -> "Satelite":

        self.radius = altitude + Satelite.earth_radius
        self.phi = phi
        self.theta = theta
        self.offset_phi = offset_phi
        self.offset_theta = offset_theta

    def getPos(self, time: float) -> tuple[float]:
        phi, theta = self.phi*time+self.offset_phi, self.theta*time+self.offset_theta

        pos = (math.cos(phi)*math.cos(theta),
               math.cos(phi)*math.sin(theta),
               math.sin(theta))

        return (self.altitude*x for x in pos)

print((i for i in range(3)))

print(Satelite.earth_radius)
