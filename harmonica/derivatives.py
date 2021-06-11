# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Compute spatial derivatives of grids
"""
from .filters import (
    derivative_easting as fft_derivative_easting,
    derivative_northing as fft_derivative_northing,
    derivative_upward as fft_derivative_upward,
)
from .filters.utils import apply_filter


def derivative_easting(grid, order=1):
    """
    Calculate the derivative of a potential field in the easting direction

    Compute the spatial derivative in the easting direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the easting derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.
    """
    return apply_filter(grid, fft_derivative_easting, order=order)


def derivative_northing(grid, order=1):
    """
    Calculate the derivative of a potential field in the northing direction

    Compute the spatial derivative in the northing direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the northing derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.
    """
    return apply_filter(grid, fft_derivative_northing, order=order)


def derivative_upward(grid, order=1):
    """
    Calculate the derivative of a potential field in the upward direction

    Compute the spatial derivative in the upward direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the upward derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.
    """
    return apply_filter(grid, fft_derivative_upward, order=order)
