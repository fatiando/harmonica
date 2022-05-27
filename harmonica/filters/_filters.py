# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Frequency domain filters meant to be applied on regular grids
"""
import numpy as np


def derivative_upward_kernel(fft_grid, order=1):
    r"""
    Filter for upward derivative in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the upward derivative. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = |\mathbf{k}| ^ n

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and :math:`n` is the order of the derivative.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the upward derivative filter in frequency
        domain.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.derivative_upward
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for upward derivative in frequency domain
    da_filter = np.sqrt(k_easting**2 + k_northing**2) ** order
    return da_filter
