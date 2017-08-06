import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import functools
from liblas import file

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

las_file='data/20597825pai4.las'   # for p in las_file:
    #     print(p.x,p.y,p.z,p.classification)
    # '''Read LAS file and create an array to hold X, Y, Z values'''
    # # Get file
    # las_file = r"E:\Testing\ground_filtered.las"
    # # Read file
f = file.File(las_file, mode='r')
    # Get number of points from header
num_points = int(f.__len__())
las_header=f.header

    # with file.File('fliter.las', mode='w', header=las_header) as outfile:
    #     for point in f:
    #         if point.classification is 13:
    #             outfile.write(point)
    # Create empty numpy array
PointsXYZIC = np.empty(shape=(num_points, 5))
    # Load all LAS points into numpy array
counter = 0
for p in f:
    newrow = [p.x, p.y, p.z, p.intensity, p.classification]
    PointsXYZIC[counter] = newrow
    counter += 1
filter_array = np.any([PointsXYZIC[:, 4] == 13],axis=0)
points=PointsXYZIC[filter_array][:1000,0:3]


fun = functools.partial(error, points=points)
params0 = [0, 0, 0]
res = scipy.optimize.minimize(fun, params0)

a = res.x[0]
b = res.x[1]
c = res.x[2]

xs, ys, zs = zip(*points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs)

point  = np.array([0.0, 0.0, c])
normal = np.array(cross([1,0,a], [0,1,b]))
d = -point.dot(normal)
xx, yy = np.meshgrid([-5,10], [-5,10])
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
ax.plot_surface(xx, yy, z, alpha=0.2, color=[0,1,0])

# ax.set_xlim(-10,10)
# ax.set_ylim(-10,10)
# ax.set_zlim(  0,10)

plt.show()