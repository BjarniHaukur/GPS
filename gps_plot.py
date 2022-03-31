import math
from Satelite import SateliteSystem
from matplotlib import pyplot as plt


def plot_satelites(sys: SateliteSystem, receiver: list[float], name="test"):
    satelites: list[SateliteSystem] = sys.get_satelites()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = [sat.A for sat in satelites]
    ys = [sat.B for sat in satelites]
    zs = [sat.C for sat in satelites]

    
    ax.set_xlim(-25000, 25000)
    ax.set_ylim(-25000, 25000)
    ax.set_zlim(-25000, 25000)
    ax.plot(0, 0, 0, marker='o', markersize=math.sqrt(SateliteSystem.earth_radius), c="blue")
    ax.plot(receiver[0], receiver[1], receiver[2], marker='o', markersize=5, c="orange")
    for x,y,z in zip(xs,ys,zs):
        ax.plot([receiver[0],x], [receiver[1],y], [receiver[2],z], c="red", linewidth=2)
    ax.scatter(xs, ys, zs, c="grey", s=100)

    for angle in range(0, 361):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.005)
        if angle%45==0:
            plt.savefig(f"myndir/{name}_{angle}")