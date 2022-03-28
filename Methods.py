import math
import numpy as np

Point = list[float, 3] # Er í raun np.array
Vector = list[Point] # -||-
e_mach = 2**-52

def distance(p: Point, p0: Point) -> float:
    return math.sqrt(sum([(x - x0)**2 for (x, x0) in zip(p, p0)]))

def Jacobi_row(p: Point, p0: Point) -> Point:
    dist = distance(p, p0)
    return (p-p0)/dist

def GaussNewton(x0: Point, centers: Vector, radii: list[float]) -> Point:
    assert len(centers) == len(radii), "Input arrays must have same length"

    x1 = np.zeros_like(x0)+2*e_mach
    while (np.any((x1 - x0) > e_mach)):
        x0 = x1
        A = np.array([Jacobi_row(x1, xy) for xy in centers])
        A_T = A.T
        r = np.array([distance(x1, xy) for xy in centers]) - radii
        v = np.linalg.solve(A_T@A, -A_T@r)
        x1 = x1 + v

    return x1