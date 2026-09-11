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
import xarray as xr


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


def get_spacing(coordinate: xr.DataArray) -> float:
    """
    Return spacing of a grid coordinate.

    Parameters
    ----------
    coordinate : xarray.DataArray
        DataArray containing the coordinate.
    coordinate : str
        Coordinate name.

    Returns
    -------
    spacing : float
    """
    spacing = coordinate.values[1] - coordinate.values[0]
    if not np.allclose(spacing, coordinate.values[1:] - coordinate.values[:-1]):
        msg = f"Invalid '{coordinate.name}' coordinates: they must be evenly spaced."
        raise ValueError(msg)
    if spacing <= 0:
        msg = (
            f"Invalid coordinate '{coordinate.name}': it must be increasingly ordered."
        )
        raise ValueError(msg)
    return spacing
