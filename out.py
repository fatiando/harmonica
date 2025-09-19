import numpy as np
import harmonica as hm
import verde as vd

h = np.array([1, 2, 3])
n = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)

n1 = n.copy()
n2 = n.copy() * 2
multi = np.array([n1, n2])

a, b, c = 5, 4, 3
x, y, z = vd.grid_coordinates(
    region=(-10, 10, -10, 10), shape=(15, 15), extra_coords=30
)
# x, y, z = 10, 10, 30

lambda_ = hm._forward.utils_ellipsoids._calculate_lambda(x, y, z, a, b, c)

hm._forward.ellipsoid_magnetics._construct_n_matrix_external(x, y, z, a, b, c, lambda_)
