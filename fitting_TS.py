import itertools
import numpy as np
from liblas import file,header,point
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import scipy.linalg
import ransac as rc
from matplotlib import cm
from matplotlib.colors import LightSource



def write_las_file(filePath,header,point):
    f = file.File('junk.las',mode='w', header= header)
    pt = point.Point()
    f.write(pt)
    f.close()


if __name__ == '__main__':
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


    PointsXYZIC = np.empty(shape=(num_points, 5))
    # Load all LAS points into numpy array
    counter = 0
    for p in f:
        # newrow = [p.x, p.y, p.z, p.intensity, p.classification]
        newrow = [p.z, p.y, p.x, p.intensity, p.classification]

        PointsXYZIC[counter] = newrow
        counter += 1

    filter_array = np.any([PointsXYZIC[:, 4] == 13],axis=0)
    PointsXYZIC_filter=PointsXYZIC[filter_array]

    data = PointsXYZIC_filter[:1000,0:3]

    rc.test()

    # # regular grid covering the domain of the data
    # mn = np.min(data,axis=0)
    # mx = np.max(data,axis=0)
    # X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    # #
    # XX = X.flatten()
    # YY = Y.flatten()
    #
    # # 43343
    # # 443862
    #
    # order = 1  # 1: linear, 2: quadratic
    # if order == 1:
    #     # best-fit linear plane
    #     A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    #     C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
    #     result=scipy.linalg.lstsq(A, data[:, 2])
    #     # evaluate it on grid
    #     Z = C[0] * X + C[1] * Y + C[2]
    #     # Z[Z > mx[2]]=mx[2]
    #     # Z[Z < mn[2]]=mn[2]
    #     # or expressed using matrix/vector product
    #     # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    #
    # elif order == 2:
    #     # best-fit quadratic curve
    #     A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    #     C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
    #
    #     # evaluate it on a grid
    #     Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
    #
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    # # ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.2)
    # # ax.scatter(X, Y, Z,c='r', s=0.1)
    # ax.scatter(data[:,0], data[:,1], data[:,2],c='r', s=0.1)
    # # ax.scatter(X, Y, c='b', s=0.1)
    #
    #
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('equal')
    # ax.axis('tight')
    #
    # plt.show()