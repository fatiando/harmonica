"""
Utility functions for equivalent layer gridders
"""
from warnings import warn
from numba import jit, prange


@jit(nopython=True, parallel=True)
def jacobian_numba(coordinates, points, jac, greens_function):
    """
    Calculate the Jacobian matrix using numba to speed things up.

    It works both for Cartesian and spherical coordiantes. We need to pass the
    corresponding Green's function through the ``greens_function`` argument.
    The Jacobian is built in parallel in order to reduce the computation time.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in prange(east.size):
        for j in range(point_east.size):
            jac[i, j] = greens_function(
                east[i],
                north[i],
                upward[i],
                point_east[j],
                point_north[j],
                point_upward[j],
            )


@jit(nopython=True, parallel=True)
def predict_numba(
    coordinates, points, coeffs, result, greens_function
):  # pylint: disable=not-an-iterable
    """
    Calculate the predicted data using numba for speeding things up.

    It works both for Cartesian and spherical coordiantes. We need to pass the
    corresponding Green's function through the ``greens_function`` argument.
    The prediction is run in parallel in order to reduce the computation time.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in prange(east.size):
        for j in range(point_east.size):
            result[i] += coeffs[j] * greens_function(
                east[i],
                north[i],
                upward[i],
                point_east[j],
                point_north[j],
                point_upward[j],
            )


def pop_extra_coords(kwargs):
    """
    Remove extra_coords from kwargs
    """
    if "extra_coords" in kwargs:
        warn("EQL gridder will ignore extra_coords: {}.".format(kwargs["extra_coords"]))
        kwargs.pop("extra_coords")
