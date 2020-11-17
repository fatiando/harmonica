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
import numba


def jacobian(
    coordinates, points, jac, greens_function
):  # pylint: disable=not-an-iterable
    """
    Calculate the Jacobian matrix

    It works both for Cartesian and spherical coordiantes. We need to pass the
    corresponding Green's function through the ``greens_function`` argument.
    The Jacobian can be built in parallel using Numba ``jit`` decorator with
    ``parallel=True``.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in numba.prange(east.size):
        for j in range(point_east.size):
            jac[i, j] = greens_function(
                east[i],
                north[i],
                upward[i],
                point_east[j],
                point_north[j],
                point_upward[j],
            )


def predict(
    coordinates, points, coeffs, result, greens_function
):  # pylint: disable=not-an-iterable
    """
    Calculate the predicted data

    It works both for Cartesian and spherical coordiantes. We need to pass the
    corresponding Green's function through the ``greens_function`` argument.
    The prediction can be run in parallel using Numba ``jit`` decorator with
    ``parallel=True``.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in numba.prange(east.size):
        for j in range(point_east.size):
            result[i] += coeffs[j] * greens_function(
                east[i],
                north[i],
                upward[i],
                point_east[j],
                point_north[j],
                point_upward[j],
            )


predict_numba_serial = numba.jit(nopython=True)(predict)
predict_numba_parallel = numba.jit(nopython=True, parallel=True)(predict)
jacobian_numba_serial = numba.jit(nopython=True)(jacobian)
jacobian_numba_parallel = numba.jit(nopython=True, parallel=True)(jacobian)


def pop_extra_coords(kwargs):
    """
    Remove extra_coords from kwargs
    """
    if "extra_coords" in kwargs:
        warn("EQL gridder will ignore extra_coords: {}.".format(kwargs["extra_coords"]))
        kwargs.pop("extra_coords")
