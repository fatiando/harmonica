# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for FFT filters
"""
import numpy as np

from ._fft import fft, ifft


def apply_filter(grid, fft_filter, **kwargs):
    """
    Apply a filter to a grid and return the transformed grid in spatial domain

    Computes the Fourier transform of the given grid, builds the filter,
    applies it and returns the inverse Fourier transform of the filtered grid.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    fft_filter : func
        Callable that builds the filter in the frequency domain.
    kwargs :
        Any additional keyword argument that should be passed to the
        ``fft_filter``.

    Returns
    -------
    filtered_grid : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the filtered version of the passed
        ``grid``. Defined are in the spatial domain.
    """
    # Catch the dims of the grid
    dims = grid.dims
    # Check if the array has two dimensions
    if len(dims) != 2:
        raise ValueError(
            f"Invalid grid with {len(dims)} dimensions. "
            + "The passed grid must be a 2 dimensional array."
        )
    # Check if the grid has nans
    if np.isnan(grid).any():
        raise ValueError(
            "Found nan(s) on the passed grid. "
            + "The grid must not have missing values before computing the "
            + "Fast Fourier Transform."
        )
    # Compute Fourier Transform of the grid
    fft_grid = fft(grid)
    # Build the filter
    da_filter = fft_filter(fft_grid, **kwargs)
    # Apply the filter
    filtered_fft_grid = fft_grid * da_filter
    # Compute inverse FFT
    filtered_grid = ifft(filtered_fft_grid).real
    return filtered_grid
