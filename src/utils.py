import numpy as np
import matplotlib.pyplot as plt

def evaluate_points(points, size_mm=-1):
    dists = []
    factor = 0

    for i in range(0, len(points)):
        j = (i+1)%len(points)
        current = points[i]
        next = points[j]
        d = distance(current, next)
        if size_mm > 0:
            factor = d/size_mm
        dists.append(d)
        print("{i:2} => {j:2}: {d:8.4f} ({factor:.2f})".format(**locals()))

    print("mean:   ", np.mean(dists))
    print("median: ", np.median(dists))
    print("std:    ", np.std(dists))
    print("max:    ", np.max(dists))
    print("min:    ", np.min(dists))
    print("first 7:", distance(points[0], points[6]))
    print("last 7: ", distance(points[27], points[34]))

def plot_3d(points, marker = 'o'):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = []
    ys = []
    zs = []
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        z = point[2]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ax.scatter(x, y, z, marker=marker)
    ax.plot(xs,ys,zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def distance(X1, X2):
    return np.abs(np.linalg.norm(X2[:3]-X1[:3]))