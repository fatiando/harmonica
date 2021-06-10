# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Filters for applying spatial derivatives in the frequency domain
"""
import numpy as np


def derivative_easting(fourier_transform, order=1):
    """
    Compute the easting derivative in the frequency domain

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    deriv_ft : :class:`xarray.DataArray`
        Array with the easting derivative of the original grid in the
        frequency domain.
    """
    # Catch the dims of the Fourier transoformed grid
    dims = fourier_transform.dims
    # Grab the easting coordinates of the Fourier transformed grid
    freq_easting = fourier_transform.coords[dims[1]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    # Compute the easting derivative in the frequency domain
    deriv_ft = fourier_transform * (k_easting * 1j) ** order
    return deriv_ft


def derivative_northing(fourier_transform, order=1):
    """
    Compute the northing derivative in the frequency domain

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    deriv_ft : :class:`xarray.DataArray`
        Array with the northing derivative of the original grid in the
        frequency domain.
    """
    # Catch the dims of the Fourier transoformed grid
    dims = fourier_transform.dims
    # Grab the northing coordinates of the Fourier transformed grid
    freq_northing = fourier_transform.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_northing = 2 * np.pi * freq_northing
    # Compute the northing derivative in the frequency domain
    deriv_ft = fourier_transform * (k_northing * 1j) ** order
    return deriv_ft


def derivative_upward(fourier_transform, order=1):
    """
    Compute the upward derivative in the frequency domain

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    deriv_ft : :class:`xarray.DataArray`
        Array with the upward derivative of the original grid in the frequency
        domain.
    """
    # Catch the dims of the Fourier transoformed grid
    dims = fourier_transform.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fourier_transform.coords[dims[1]]
    freq_northing = fourier_transform.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the upward (vertical) derivative in the frequency domain
    deriv_ft = fourier_transform * np.sqrt(k_easting ** 2 + k_northing ** 2) ** order
    return deriv_ft
