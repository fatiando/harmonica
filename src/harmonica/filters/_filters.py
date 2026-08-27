# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Frequency domain filters meant to be applied on regular grids.
"""

import numpy as np

from .._utils import magnetic_angles_to_vec
from . import _padding
from ._fft import fft, ifft
from ._utils import grid_sanity_checks


def apply_filter(
    grid,
    fft_filter,
    *,
    filter_kwargs=None,
    pad=True,
    pad_kwargs=None,
    drop_coords=False,
):
    """
    Apply a filter to a grid and return the transformed grid in spatial domain.

    Computes the Fourier transform of the given grid, builds the filter,
    applies it and returns the inverse Fourier transform of the filtered grid.

    .. note::

        Any non-dimensional coordinates in the original grid will be dropped
        from the filtered grid. This is because we can't know if the filter
        invalidates the coordinate values (for example, upward continuation
        would invalidate any height coordinates). So it's safer to drop them.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    fft_filter : func
        Callable that builds the filter in the frequency domain.
    filter_kwargs : dict or None, optional
        Any additional keyword argument that should be passed to the
        ``fft_filter`` in the form of a dictionary.
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function. If none are given, the default
        padding of 25% the dimensions of the grid will be added using the
        "edge" method.
    drop_coords : bool, optional
        If True, non-dimensional coordinates of the grid will be dropped after
        filtering. This is useful if the filter could move the grid, like in upward
        continuation, which could make these coordinates incorrect.

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

    # TODO: Fix this in our own FFT functions
    # Need to remove non-dimensional coordinates before padding and FFT because
    # xrft doesn't know what to do with them.
    non_dim_coords = {c: grid[c] for c in grid.coords if c not in grid.indexes}
    grid = grid.drop_vars(non_dim_coords.keys())

    if pad:
        # By default, use a padding width of 25% of each grid dimension.
        # Fedi et al. (2012; doi:10.1111/j.1365-246X.2011.05259.x) suggest
        # a padding of 100% but that seems exaggerated.
        if "pad_width" not in pad_kwargs:
            pad_kwargs["pad_width"] = {d: int(0.25 * grid[d].size) for d in dims}
        if "mode" not in pad_kwargs:
            pad_kwargs["mode"] = "edge"
        if "constant_values" not in pad_kwargs:
            # Has to be included explicitly as None since the pad function always
            # passes it to xarray.DataArray.pad.
            pad_kwargs["constant_values"] = None
        fft_grid = fft(_padding.pad(grid, **pad_kwargs))
    else:
        fft_grid = fft(grid)

    # The filter convolution in the frequency domain is a multiplication
    filtered_fft_grid = fft_grid * fft_filter(fft_grid, **filter_kwargs)

    # Keep only the real part since the inverse transform returns complex
    # number by default
    filtered_grid = ifft(filtered_fft_grid).real
    if pad:
        filtered_grid = _padding.unpad(filtered_grid, pad_kwargs["pad_width"])

    # Restore the original coordinates to the grid because the inverse
    # transform calculates coordinates from the frequencies, which can lead to
    # rounding errors and coordinates that are slightly off. This causes errors
    # when doing operations with the transformed grids. Restoring the original
    # coordinates avoids these issues.
    filtered_grid = filtered_grid.assign_coords(
        {dims[1]: grid[dims[1]], dims[0]: grid[dims[0]]}
    )

    # Restore the non-dimensional coordinates if desired
    if not drop_coords:
        filtered_grid = filtered_grid.assign_coords(
            {name: non_dim_coords[name] for name in non_dim_coords}
        )

    return filtered_grid


def derivative_upward_kernel(fft_grid, order=1):
    r"""
    Filter for upward derivative in frequency domain.

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
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
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

    See Also
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
    da_filter = (-np.sqrt(k_easting**2 + k_northing**2)) ** order
    return da_filter


def derivative_easting_kernel(fft_grid, order=1):
    r"""
    Filter for easting derivative in frequency domain.

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the easting derivative. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = (i k_e)^n

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector), :math:`k_e` is the easting wavenumber component of
    :math:`\mathbf{k}`, :math:`i` is the imaginary unit and :math:`n` is the
    order of the derivative.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the easting derivative filter in frequency
        domain.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.derivative_easting
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    # Compute the filter for easting derivative in frequency domain
    da_filter = (k_easting * 1j) ** order
    return da_filter


def derivative_northing_kernel(fft_grid, order=1):
    r"""
    Filter for northing derivative in frequency domain.

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the northing derivative. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = (i k_n)^n

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector), :math:`k_n` is the northing wavenumber component of
    :math:`\mathbf{k}`, :math:`i` is the imaginary unit and :math:`n` is the
    order of the derivative.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the northing derivative filter in frequency
        domain.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.derivative_northing
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for northing derivative in frequency domain
    da_filter = (k_northing * 1j) ** order
    return da_filter


def upward_continuation_kernel(fft_grid, height_displacement):
    r"""
    Filter for upward continuation in frequency domain.

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the upward continuation. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = e^{-|\mathbf{k}| \Delta h}

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and :math:`\Delta h` is the height displacement of the
    upward continuation.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
    height_displacement : float
        The height displacement of upward continuation. For upward
        continuation, the height displacement should be positive.
        It has the same units as the input xarray data coordinates.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the upward continuation filter in frequency
        domain.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.upward_continuation
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for upward continuation in frequency domain
    da_filter = np.exp(-np.sqrt(k_easting**2 + k_northing**2) * height_displacement)
    return da_filter


def gaussian_lowpass_kernel(fft_grid, wavelength):
    r"""
    Filter for Gaussian low-pass in frequency domain.

    Return a :class:`xarray.DataArray` with the values of a Gaussian low-pass
    filter the frequency domain. The filter is built upon the frequency
    coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) =
            e^{
                - \frac{1}{2} \left( \frac{|\mathbf{k}|}{k_c} \right)^2
            }

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and :math:`k_c` is the cutoff wavenumber:
    :math:`k_c = \frac{2\pi}{\lambda_c}`,
    where :math:`\lambda_c` is the cutoff wavelength.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
    wavelength : float
        The cutoff wavelength for the low-pass filter.
        Its units should be the inverse units of the coordinates in
        ``fft_grid``.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the Gaussian low-pass filter in frequency
        domain.

    References
    ----------
    [Geosoft1999]_

    See Also
    --------
    harmonica.gaussian_lowpass
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for northing derivative in frequency domain
    da_filter = np.exp(
        -(k_easting**2 + k_northing**2) / (2 * (2 * np.pi / wavelength) ** 2)
    )
    return da_filter


def gaussian_highpass_kernel(fft_grid, wavelength):
    r"""
    Filter for Gaussian high-pass in frequency domain.

    Return a :class:`xarray.DataArray` with the values of a Gaussian high-pass
    filter the frequency domain. The filter is built upon the frequency
    coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) =
            1 - e^{
                - \frac{1}{2} \left( \frac{|\mathbf{k}|}{k_c} \right)^2
            }

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and :math:`k_c` is the cutoff wavenumber:
    :math:`k_c = \frac{2\pi}{\lambda_c}`,
    where :math:`\lambda_c` is the cutoff wavelength.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
    wavelength : float
        The cutoff wavelength for the high-pass filter.
        Its units should be the inverse units of the coordinates in
        ``fft_grid``.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the Gaussian high-pass filter in frequency
        domain.

    References
    ----------
    [Geosoft1999]_

    See Also
    --------
    harmonica.gaussian_highpass
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for northing derivative in frequency domain
    da_filter = 1 - np.exp(
        -(k_easting**2 + k_northing**2) / (2 * (2 * np.pi / wavelength) ** 2)
    )
    return da_filter


def reduction_to_pole_kernel(
    fft_grid,
    inclination,
    declination,
    magnetization_inclination,
    magnetization_declination,
):
    r"""
    Filter for reduction to the pole in the frequency domain.

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for applying a reduction to the pole on magnetic data. The filter
    is built upon the frequency coordinates of the passed ``fft_grid`` and is
    defined as follows:

    .. math::

        g(\mathbf{k}) = \frac{1}{\Theta_m \Theta_f}

    with

    .. math::

        \Theta_m = m_z + i \frac{m_e k_e + m_n k_n}{|\mathbf{k}|}

    .. math::

        \Theta_f = f_z + i \frac{f_e k_e + f_n k_n}{|\mathbf{k}|}

    where :math:`\mathbf{k} = (k_e, k_n)` is the wavenumber vector,
    :math:`\hat{\mathbf{f}} = (f_e, f_n, f_z)` is a unit vector parallel
    to the geomagnetic field and :math:`\hat{\mathbf{m}} = (m_e, m_n, m_z)`
    is a unit vector parallel to the magnetization vector of the source. The
    :math:`f_e`, :math:`f_n`, :math:`m_e`, :math:`m_n` are the easting and
    northing components while the :math:`f_z` and :math:`m_z` are the
    **downward** components.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`harmonica.filters.fft` and :func:`harmonica.filters.ifft` functions
        to compute the Fourier Transform and its inverse, respectively.
    inclination : float in degrees
        The inclination of the inducing Geomagnetic field.
    declination : float in degrees
        The declination of the inducing Geomagnetic field.
    magnetization_inclination : float in degrees
        The inclination of the total magnetization of the anomaly source.
    magnetization_declination : float in degrees
        The declination of the total magnetization of the anomaly source.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the reduction to the pole filter in frequency
        domain.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.reduction_to_pole
    """
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Convert inclination and declination to versor components
    m_e, m_n, m_u = magnetic_angles_to_vec(
        1, magnetization_inclination, magnetization_declination
    )
    f_e, f_n, f_u = magnetic_angles_to_vec(1, inclination, declination)
    # Convert the upward components to downward components because the
    # equations below for the filter use downward instead
    m_z = -m_u
    f_z = -f_u
    # Compute the filter for reduction to pole in frequency domain
    k_squared = k_northing**2 + k_easting**2
    k = np.sqrt(k_squared)
    # Compute the rtp filter
    da_filter = (
        k_squared
        * (f_z * k + 1j * (f_e * k_easting + f_n * k_northing)) ** (-1)
        * (m_z * k + 1j * (m_e * k_easting + m_n * k_northing)) ** (-1)
    )
    # Set 0 wavenumber to 0
    da_filter.loc[{dims[0]: 0, dims[1]: 0}] = 0
    return da_filter
