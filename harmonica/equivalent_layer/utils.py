# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for equivalent layer gridders
"""
from warnings import warn
from numba import jit, prange


@jit(nopython=True)
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


@jit(nopython=True)
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


# Define parallelized version of the utils functions
if hasattr(predict_numba, "py_func"):
    predict_numba_parallel = jit(nopython=True, parallel=True)(predict_numba.py_func)
else:
    predict_numba_parallel = jit(nopython=True, parallel=True)(predict_numba)
if hasattr(jacobian_numba, "py_func"):
    jacobian_numba_parallel = jit(nopython=True, parallel=True)(jacobian_numba.py_func)
else:
    jacobian_numba_parallel = jit(nopython=True, parallel=True)(jacobian_numba)


def dispatch_predict(parallel):
    """
    Return either the parallel or non-parallel predict function
    """
    if parallel:
        return predict_numba_parallel
    else:
        return predict_numba


def dispatch_jacobian(parallel):
    """
    Return either the parallel or non-parallel jacobian function
    """
    if parallel:
        return jacobian_numba_parallel
    else:
        return jacobian_numba


def pop_extra_coords(kwargs):
    """
    Remove extra_coords from kwargs
    """
    if "extra_coords" in kwargs:
        warn("EQL gridder will ignore extra_coords: {}.".format(kwargs["extra_coords"]))
        kwargs.pop("extra_coords")
