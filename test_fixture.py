import math
import numpy as np
from Satelite import SateliteConnection, SateliteSystem
from Methods import angle_to_coordinates

class TestGps:

    def __init__(self, receiver):
        self.receiver: tuple[float] = receiver

        self.__quadrant_phi = math.floor(receiver[0]/(math.pi/2))
        self.__quadrant_theta = math.floor(receiver[1]/(math.pi/2))
        self.__initial_guess = ((self.__quadrant_phi+0.5)*math.pi/2, (self.__quadrant_theta+0.5)*math.pi/2)

    def get_receiver_pos(self) -> tuple[float]:
        return angle_to_coordinates(
            SateliteSystem.earth_radius, *self.receiver
        )

    def get_initial_guess(self) -> tuple[float]:
        """ Returns an initial guess in the middle of the section where the request came from """
        return self.__initial_guess

    def get_satelites(self, rho = 26570, phi_diff = math.pi/2, theta_diff = 2*math.pi, n = 4, d = 0.0001) -> list[SateliteConnection]:
        receiver_pos = self.get_receiver_pos()
        phi, theta = self.__initial_guess()

        phi_values = np.random.uniform(phi-phi_diff, phi+phi_diff, n)
        theta_values = np.random.uniform(theta-theta_diff, theta+theta_diff, n)

        return [self.make_satelite(rho, p, t, receiver_pos, d) for (p,t) in zip(phi_values, theta_values)]

    def make_satelite(self, rho, phi, theta, pos, d) -> SateliteConnection:
        A, B, C = angle_to_coordinates(rho, phi, theta)
        x, y, z = pos
        R = math.sqrt((A-x)**2 + (B-y)**2 + (C-z)**2)
        t = d + R/SateliteSystem.speed_of_light
        return SateliteConnection(A, B, C, t)


# receiver = (math.pi*2, math.pi/2)
# test = TestGps(receiver)

# print(receiver)
# init = test.get_initial_guess()
# print(init)
# print(test.angle_to_coordinates(22000, *init))