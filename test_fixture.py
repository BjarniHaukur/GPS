import math
import numpy as np
from Satelite import SateliteConnection, SateliteSystem

class TestGps:

    def __init__(self, receiver, num_satelites):
        self.receiver: tuple[float] = receiver
        self.num_satelites: int = num_satelites

        # initial_guess: tuple[float] = 

    def get_receiver_pos(self):
        return self.angle_to_coordinates(
            SateliteSystem.earth_radius, self.receiver[0], self.receiver[1]
        )

    def angle_to_coordinates(self, rho, phi, theta):
        return (rho*math.cos(phi)*math.cos(theta),
                rho*math.cos(phi)*math.sin(theta),
                rho*math.sin(phi)
        )


    def get_initial_guess(self):
        phi, theta = self.receiver
        quadrant_phi = math.floor(phi/(math.pi/2))
        quadrant_theta = math.floor(theta/(math.pi/2))
        return ((quadrant_phi+0.5)*math.pi/2, (quadrant_theta+0.5)*math.pi/2)


test = TestGps((math.pi, 0.8*math.pi), 10)
print(test.get_initial_guess())