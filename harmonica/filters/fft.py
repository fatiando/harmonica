# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Wrap xrft functions to compute FFTs and inverse FFTs
"""

from xrft.xrft import fft as _fft, ifft as _ifft


def fft(grid, **kwargs):
    """
    Compute Fast Fourier Transform of a 2D regular grid

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.

    Returns
    -------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
    """
    return _fft(grid, **kwargs)


def ifft(fourier_transform, easting_shift=None, northing_shift=None, **kwargs):
    """
    Compute Inverse Fast Fourier Transform of a 2D regular grid

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with a regular grid defined in the frequency domain.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
    easting_shift : float or None (optional)
        Minimum value of the easting coordinates of the original grid in the
        spatial domain.
        This value is used to shift the easting coordinate of the ifft grid,
        otherwise :func:`xrft.xrft.ifft` will center them around zero.
        If None, the easting coordinates will be centered around zero.
        Defaults to None.
    northing_shift : float or None (optional)
        Minimum value of the northing coordinates of the original grid in the
        spatial domain.
        This value is used to shift the northing coordinate of the ifft grid,
        otherwise :func:`xrft.xrft.ifft` will center them around zero.
        If None, the northing coordinates will be centered around zero.
        Defaults to None.

    Returns
    -------
    grid : :class:`xarray.DataArray`
        Array with the inverse Fourier transform of the passed grid.
    """
    grid = _ifft(fourier_transform, **kwargs)
    # Move recovered coordinates to original range
    if easting_shift is not None or northing_shift is not None:
        coords = {dim: grid.coords[dim] for dim in grid.dims}
        if easting_shift is not None:
            # Grab the easting dimension name from grid.dims
            dim = grid.dims[1]
            # Define a new easting dimension by shifting the one in grid
            easting = easting_shift + (
                grid.coords[dim].values - grid.coords[dim].values.min()
            )
            coords[dim] = easting
        if northing_shift is not None:
            # Grab the northing dimension name from grid.dims
            dim = grid.dims[0]
            # Define a new northing dimension by shifting the one in grid
            northing = northing_shift + (
                grid.coords[dim].values - grid.coords[dim].values.min()
            )
            coords[dim] = northing
        grid = grid.assign_coords(coords)
    return grid
