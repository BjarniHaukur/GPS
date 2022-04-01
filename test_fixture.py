import math
import numpy as np
from typing import Callable
from Satelite import SateliteConnection, SateliteSystem, DynamicSystem
from Methods import angle_to_coordinates
import pandas as pd

class TestGps:

    def __init__(self, receiver, d = 0.0001):
        self.receiver: tuple[float] = receiver
        self.d = d

        self.__quadrant_phi = math.floor(receiver[0]/(math.pi/2))
        self.__quadrant_theta = math.floor(receiver[1]/(math.pi/2))
        self.__initial_guess = ((self.__quadrant_phi+0.5)*math.pi/2, (self.__quadrant_theta+0.5)*math.pi/2)

    def get_receiver_pos(self) -> tuple[float]:
        return np.array(angle_to_coordinates(
            SateliteSystem.earth_radius, *self.receiver
        ))

    def get_initial_guess(self) -> tuple[float]:
        """ Returns an initial guess in the middle of the section where the request came from """
        return np.array([*angle_to_coordinates(SateliteSystem.earth_radius, *self.__initial_guess), self.d])

    def get_random_satelites(self, phi_diff = math.pi/2, theta_diff = 2*math.pi, rho = 26570, n = 4) -> list[SateliteConnection]:
        receiver_pos = self.get_receiver_pos()
        phi, theta = self.__initial_guess

        phi_values = np.random.uniform(phi-phi_diff, phi+phi_diff, n)
        theta_values = np.random.uniform(theta-theta_diff, theta+theta_diff, n)

        return [self.make_satelite(rho, p, t, receiver_pos) for (p,t) in zip(phi_values, theta_values)]

    def get_linspace_satelites(self, phi_diff = math.pi/2, theta_diff = 2*math.pi, rho = 26570, n = 4) -> list[SateliteConnection]:
        receiver_pos = self.get_receiver_pos()
        phi, theta = self.__initial_guess

        phi_values = np.linspace(phi-phi_diff, phi+phi_diff, n)
        theta_values = np.linspace(theta-theta_diff, theta+theta_diff, n)

        return [self.make_satelite(rho, p, t, receiver_pos) for (p,t) in zip(phi_values, theta_values)]

    def make_satelite(self, rho, phi, theta, pos) -> SateliteConnection:
        A, B, C = angle_to_coordinates(rho, phi, theta)
        x, y, z = pos
        R = math.sqrt((A-x)**2 + (B-y)**2 + (C-z)**2)
        t = self.d + R/SateliteSystem.speed_of_light
        return SateliteConnection(A, B, C, t)


SateliteGenerator = Callable[[], list[SateliteConnection]]

def run_tests(test_fixt: TestGps, sat_gen: SateliteGenerator, n_in: int = 50, n_out: int = 10, t_err_min: float = 10**(-12), t_err_max: float = 10**(-8)) -> pd.DataFrame:
    df = pd.DataFrame(index=list(range(n_out)), columns=["min_pos_error", "avg_pos_error", "max_pos_error", "condition_number"])
    guess = test_fixt.get_initial_guess()
    for i in range(n_out):
        
        ds = DynamicSystem(sat_gen())
        pe_hist = np.zeros(n_in)
        emf_hist = np.zeros(n_in)

        for j in range(n_in):
            pe_hist[j], emf_hist[j] = ds.compute_EMF(guess, t_err_min, t_err_max)
        
        df.at[i, "min_pos_error"] = np.amin(pe_hist)*1000
        df.at[i, "avg_pos_error"] = np.average(pe_hist)*1000
        df.at[i, "max_pos_error"] = np.amax(pe_hist)*1000
        df.at[i, "condition_number"] = np.amax(emf_hist)

    return df