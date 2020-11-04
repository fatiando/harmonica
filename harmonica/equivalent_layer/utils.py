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
from numba import jit


@jit(nopython=True)
def jacobian_numba(coordinates, points, jac, greens_function):
    """
    Calculate the Jacobian matrix using numba to speed things up.

    It works both for Cartesian and spherical coordiantes.
    We need to pass the corresponding Green's function through the
    ``greens_function`` argument.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in range(east.size):
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
def predict_numba(coordinates, points, coeffs, result, greens_function):
    """
    Calculate the predicted data using numba for speeding things up.

    It works both for Cartesian and spherical coordiantes.
    We need to pass the corresponding Green's function through the
    ``greens_function`` argument.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in range(east.size):
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
