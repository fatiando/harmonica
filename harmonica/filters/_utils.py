# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for FFT filters.
"""

import numpy as np
import xrft

from ._fft import fft, ifft


def apply_filter(grid, fft_filter, filter_kwargs=None, pad_kwargs=None):
    """
    Apply a filter to a grid and return the transformed grid in spatial domain.

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
    if filter_kwargs is None:
        filter_kwargs = {}
    if pad_kwargs is None:
        pad_kwargs = {}
    grid_sanity_checks(grid)
    dims = grid.dims
    # Need to remove non-dimensional coordinates before padding and FFT because
    # xrft doesn't know what to do with them.
    non_dim_coords = {c: grid[c] for c in grid.coords if c not in grid.indexes}
    grid = grid.drop_vars(non_dim_coords.keys())
    # By default, use a padding width of 25% of the largest grid dimension.
    # Fedi et al. (2012; doi:10.1111/j.1365-246X.2011.05259.x) suggest
    # a padding of 100% but that seems exaggerated.
    if "pad_width" not in pad_kwargs:
        width = int(0.25 * max(grid[d].size for d in dims))
        pad_kwargs["pad_width"] = {d: width for d in dims}
    if "mode" not in pad_kwargs:
        pad_kwargs["mode"] = "edge"
        # Has to be included explicitly as None or numpy complains about the
        # argument being there.
        pad_kwargs["constant_values"] = None
    fft_grid = fft(xrft.pad(grid, **pad_kwargs))
    # The filter convolution in the frequency domain is a multiplication
    filtered_fft_grid = fft_grid * fft_filter(fft_grid, **filter_kwargs)
    # Keep only the real part since the inverse transform returns complex
    # number by default
    filtered_grid = xrft.unpad(ifft(filtered_fft_grid).real, pad_kwargs["pad_width"])
    # Restore the original coordinates to the grid because the inverse
    # transform calculates coordinates from the frequencies, which can lead to
    # rounding errors and coordinates that are slightly off. This causes errors
    # when doing operations with the transformed grids. Restoring the original
    # coordinates avoids these issues.
    filtered_grid = filtered_grid.assign_coords(
        {dims[1]: grid[dims[1]].values, dims[0]: grid[dims[0]].values}
    )
    return filtered_grid


def grid_sanity_checks(grid):
    """
    Run sanity checks on the grid.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.

    Raises
    ------
    ValueError
        If the passed grid is not 2D or if it contains nan values.
    """
    # Check if the array has two dimensions
    if (n_dims := len(grid.dims)) != 2:
        raise ValueError(
            f"Invalid grid with {n_dims} dimensions. "
            + "The passed grid must be a 2 dimensional array."
        )
    # Check if the grid has nans
    if np.isnan(grid).any():
        raise ValueError(
            "Found nan(s) on the passed grid. "
            + "The grid must not have missing values before computing the "
            + "Fast Fourier Transform."
        )
