"""
Equivalent Layer interpolators for harmonic functions
"""
import numpy as np
from numba import jit
from sklearn.utils.validation import check_is_fitted
from verde import get_region
from verde.base import BaseGridder, check_fit_input, least_squares
# Would use n_1d_arrays from verde.base when a new release is made


class HarmonicEQL(BaseGridder):
    """
    3D Equivalent Layer interpolator for harmonic fields using Green's functions
    """

    def __init__(self, damping=None):
        """
        """
        self.damping = damping

    def fit(self, coordinates, data, weights=None, points=None, depth=-1e3):
        """
        """
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
        if points is None:
            # Define a default set of point masses. This is not intended to be on the
            # final version.
            self.points = [np.atleast_1d(i).ravel().copy() for i in coordinates]
            self.points[-1] += depth
        else:
            self.points = [np.atleast_1d(i).ravel() for i in points[:3]]
        jacobian = self.jacobian(coordinates)
        self.masses_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """
        """
        # We know the gridder has been fitted if it has the masses_
        check_is_fitted(self, ["masses_"])
        shape = np.broadcast(*coordinates[:3]).shape
        size = np.broadcast(*coordinates[:3]).size
        dtype = coordinates[0].dtype
        coordinates = [np.atleast_1d(i).ravel() for i in coordinates[:3]]
        data = np.zeros(size, dtype=dtype)
        predict_numba(coordinates, self.points, self.masses_, data)
        return data.reshape(shape)

    def jacobian(self, coordinates):
        """
        """
        n_coordinates = coordinates[0].size
        jac = np.zeros((n_coordinates, n_coordinates))
        jacobian_numba(*coordinates, *self.points, jac)
        return jac


@jit(nopython=True)
def predict_numba(coordinates, points, masses, result):
    """
    """
    east, north, vertical = coordinates[:]
    point_east, point_north, point_vertical = points[:]
    for i in range(east.size):
        for j in range(point_east.size):
            result[i] += masses[j] * greens_func(
                east[i],
                north[i],
                vertical[i],
                point_east[j],
                point_north[j],
                point_vertical[j],
            )


@jit(nopython=True)
def greens_func(east, north, vertical, point_east, point_north, point_vertical):
    """
    """
    distance = np.sqrt(
        (east - point_east) ** 2
        + (north - point_north) ** 2
        + (vertical - point_vertical) ** 2
    )
    return 1 / distance


@jit(nopython=True)
def jacobian_numba(east, north, vertical, point_east, point_north, point_vertical, jac):
    """
    """
    for i in range(east.size):
        for j in range(point_east.size):
            jac[i, j] = greens_func(
                east[i],
                north[i],
                vertical[i],
                point_east[j],
                point_north[j],
                point_vertical[j],
            )
