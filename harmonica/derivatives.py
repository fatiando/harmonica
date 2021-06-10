# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Compute spatial derivatives of grids
"""
import numpy as np

from .filters import (
    derivative_easting as fft_derivative_easting,
    derivative_northing as fft_derivative_northing,
    derivative_upward as fft_derivative_upward,
)
from .filters.fft import fft, ifft


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
    return _derivative_fft(grid, fft_derivative_easting, order)


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
    return _derivative_fft(grid, fft_derivative_northing, order)


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
    return _derivative_fft(grid, fft_derivative_upward, order)


def _derivative_fft(grid, fft_filter, order):
    """
    Calculate the derivative of a potential field in a particular direction

    Use the passed kernel to obtain the derivative along the chosen direction
    (easting, northing or upward).

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    fft_filter : func
        Callable that applies a filter in the frequency domain, corresponding
        to the desired direction of the derivative.
    order : int
        The order of the derivative.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the chosen directinoal derivative of
        the passed ``grid``. Its units are the same units of the ``grid`` per
        units of its coordinates.
    """
    # Catch the dims of the grid
    dims = grid.dims
    # Check if the array has two dimensions
    if len(dims) != 2:
        raise ValueError(
            f"Invalid grid with {len(dims)} dimensions. "
            + "The passed grid must be a 2 dimensional array."
        )
    # Check if grid and coordinates has nans
    if np.isnan(grid).any():
        raise ValueError(
            "Found nan(s) on the passed grid. "
            + "The grid must not have missing values before computing the "
            + "Fast Fourier Transform."
        )
    # Compute Fourier Transform of the grid
    fourier_transform = fft(grid)
    # Compute the derivative in the frequency domain
    deriv_ft = fft_filter(fourier_transform, order)
    # Compute inverse FFT
    deriv = ifft(
        deriv_ft,
        easting_shift=grid.easting.values.min(),
        northing_shift=grid.northing.values.min(),
    ).real
    return deriv
