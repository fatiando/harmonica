# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for equivalent sources gridders
"""
from warnings import warn

from numba import jit, prange


def cast_fit_input(coordinates, data, weights, dtype):
    """
    Cast the inputs of the fit method to the given dtype

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...).
    data : array
        The data values of each data point.
    weights : None or array
        If not None, then the weights assigned to each data point.

    Returns
    -------
    casted_inputs
        The casted inputs in the same order.
    """
    coordinates = tuple(c.astype(dtype) for c in coordinates)
    data = data.astype(dtype)
    if weights is not None:
        weights = weights.astype(dtype)
    return coordinates, data, weights


def pop_extra_coords(kwargs):
    """
    Remove extra_coords from kwargs
    """
    if "extra_coords" in kwargs:
        warn("EQL gridder will ignore extra_coords: {}.".format(kwargs["extra_coords"]))
        kwargs.pop("extra_coords")


def jacobian(coordinates, points, jac, greens_function):
    """
    Calculate the Jacobian matrix

    It works both for Cartesian and spherical coordinates. We need to pass the
    corresponding Green's function through the ``greens_function`` argument.
    The Jacobian can be built in parallel using Numba ``jit`` decorator with
    ``parallel=True``.
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


def predict(coordinates, points, coeffs, result, greens_function):
    """
    Calculate the predicted data

    It works both for Cartesian and spherical coordinates. We need to pass the
    corresponding Green's function through the ``greens_function`` argument.
    The prediction can be run in parallel using Numba ``jit`` decorator with
    ``parallel=True``.
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


predict_numba_serial = jit(nopython=True)(predict)
predict_numba_parallel = jit(nopython=True, parallel=True)(predict)
jacobian_numba_serial = jit(nopython=True)(jacobian)
jacobian_numba_parallel = jit(nopython=True, parallel=True)(jacobian)
