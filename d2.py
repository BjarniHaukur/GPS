import math
from turtle import pos
import numpy as np
from numpy.linalg import det
from Satelite import SateliteConnection, SateliteSystem


def row(a: SateliteConnection, b: SateliteConnection):
    xyzd = np.array([a.A-b.A, a.B-b.B, a.C-b.C, (b.t-a.t)*SateliteSystem.speed_of_light**2])*(-2)
    plus = sum([x**2 for x in [a.A, a.B, a.C, b.t*SateliteSystem.speed_of_light]])
    minus = sum([x**2 for x in [b.A, b.B, b.C, a.t*SateliteSystem.speed_of_light]])
    w = np.array([plus-minus])

    return np.hstack([xyzd, w])


if __name__ == '__main__':
    sat1 = SateliteConnection(15600,7540,20140,0.07074)
    sat2 = SateliteConnection(18760,2750,18610,0.07220)
    sat3 = SateliteConnection(17610,14630,13480,0.07690)
    sat4 = SateliteConnection(19170,610,18390,0.07242)


    matrix = np.vstack([row(sat1, x) for x in [sat2, sat3, sat4]])

    # x = i*d + j
    # y = k*d + l
    # z = m*d + n

    # regla Cramer's, sleppt mínus þegar við breyttum röðinni af dálkum
    i = -1*det(matrix[:, [1,2,3]])/det(matrix[:, [0,1,2]])
    j = -1*det(matrix[:, [1,2,4]])/det(matrix[:, [0,1,2]])
    k =    det(matrix[:, [0,2,3]])/det(matrix[:, [0,1,2]])
    l =    det(matrix[:, [0,2,4]])/det(matrix[:, [0,1,2]])
    m = -1*det(matrix[:, [0,1,3]])/det(matrix[:, [0,1,2]])
    n = -1*det(matrix[:, [0,1,4]])/det(matrix[:, [0,1,2]])

    # (id+j − A1)**2 + (kd+l − B1)**2 + (md+n − C1)**2 = (c(t1 − d))**2
    # <=>
    #    (i**2 + k**2 + m**2 - c**2)d**2 
    # + 2(ij - Ai + kl - Bk + mn -Cm + c**2t)d
    # +  (A**2 - 2Aj + j**2 + B**2 - 2Bl + l**2 + C**2 - 2Cn + n**2 -c**2t**2)
    cc = SateliteSystem.speed_of_light**2
    a = i**2 + k**2 + m**2 - cc
    b = 2*(i*j - sat1.A*i + k*l - sat1.B*k + m*n - sat1.C*m + cc*sat1.t)
    c = sat1.A**2 - 2*sat1.A*j + j**2 + sat1.B**2 - 2*sat1.B*l + k**2 +sat1.C**2 - 2*sat1.C*n + n**2 - cc*sat1.t**2

    d = b**2-4*a*c
    if d >= 0:
        r1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
        r2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)
        print("The roots are: ")
        print(f"r1 = {r1:.6f}")
        print(f"r2 = {r2:.6f}")

        pos1 = i*r1+j, k*r1+l, m*r1+n
        pos2 = i*r2+j, k*r2+l, m*r2+n

        print("By inserting the roots into one of our original equations we get: ")
        print(f"r1: {math.sqrt(sum([x**2 for x in pos1])):.2f}km")
        print(f"r2: {math.sqrt(sum([x**2 for x in pos2])):.2f}km")
        print("So the former root must be our solution since the radius of the latter is not on the Earth")