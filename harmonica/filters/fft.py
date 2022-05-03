# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Wrap xrft functions to compute FFTs and inverse FFTs
"""

from xrft.xrft import fft as _fft
from xrft.xrft import ifft as _ifft


def fft(grid, true_phase=True, true_amplitude=True, drop_bad_coords=True, **kwargs):
    """
    Compute Fast Fourier Transform of a 2D regular grid

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    true_phase : bool (optional)
        Take the coordinates into consideration, keeping the original phase of
        the coordinates in the spatial domain (``direct_lag``) and multiplies
        the FFT with an exponential function corresponding to this phase.
        Defaults to True.
    true_amplitude : bool (optional)
        If True, the FFT is multiplied by the spacing of the transformed
        variables to match theoretical FT amplitude.
        Defaults to True.
    drop_bad_coords : bool (optional)
        If True, only the indexes of the array will be kept before passing it
        to :func:`xrft.fft`. Any extra coordinate should be drooped, otherwise
        :func:`xrft.fft` raises an error after finding *bad coordinates*.
        Defaults to True.

    Returns
    -------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
    """
    if drop_bad_coords:
        bad_coords = tuple(c for c in grid.coords if c not in grid.indexes)
        grid = grid.drop(bad_coords)
    return _fft(grid, true_phase=true_phase, true_amplitude=true_amplitude, **kwargs)


def ifft(fourier_transform, true_phase=True, true_amplitude=True, **kwargs):
    """
    Compute Inverse Fast Fourier Transform of a 2D regular grid

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with a regular grid defined in the frequency domain.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
    true_phase : bool (optional)
        Take the coordinates into consideration, recovering the original
        coordinates in the spatial domain returning to the the original phase
        (``direct_lag``), and multiplies the iFFT with an exponential function
        corresponding to this phase.
        Defaults to True.
    true_amplitude : bool (optional)
        If True, output is divided by the spacing of the transformed variables
        to match theoretical IFT amplitude.
        Defaults to True.

    Returns
    -------
    grid : :class:`xarray.DataArray`
        Array with the inverse Fourier transform of the passed grid.
    """
    return _ifft(
        fourier_transform,
        true_phase=true_phase,
        true_amplitude=true_amplitude,
        **kwargs
    )
