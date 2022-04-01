import math
import numpy as np
from matplotlib import pyplot as plt




Point = list[float, 3] # Er í raun np.array
Vector = list[Point] # -||-
e_mach = 2**-52 


def angle_to_coordinates(rho, phi, theta):
    """ Turns angle and radius into sperical coordinates """
    return (rho*math.cos(phi)*math.cos(theta),
            rho*math.cos(phi)*math.sin(theta),
            rho*math.sin(phi)
    )

def distance(p: Point, p0: Point) -> float:
    return math.sqrt(sum([(x - x0)**2 for (x, x0) in zip(p, p0)]))  

def Jacobi_row(p: Point, p0: Point) -> Point:
    dist = distance(p, p0)
    return (p-p0)/dist  
    


    
