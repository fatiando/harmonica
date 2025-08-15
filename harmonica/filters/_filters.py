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
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
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
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
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
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
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
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
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
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
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
    magnetization_inclination=None,
    magnetization_declination=None,
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
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    inclination : float in degrees
        The inclination of the inducing Geomagnetic field.
    declination : float in degrees
        The declination of the inducing Geomagnetic field.
    magnetization_inclination : float in degrees or None
        The inclination of the total magnetization of the anomaly source. If
        None, the ``magnetization_inclination`` will be set equal to the
        ``inclination``, neglecting remanent magnetization and self
        demagnetization. Default None.
    magnetization_declination : float in degrees
        The declination of the total magnetization of the anomaly source. If
        None, the ``magnetization_declination`` will be set equal to the
        ``declination``, neglecting remanent magnetization and self
        demagnetization. Default None.

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
    # Check if magnetization angles are valid
    _check_magnetization_angles(magnetization_inclination, magnetization_declination)
    # Define magnetization angles if they are None
    if magnetization_declination is None and magnetization_inclination is None:
        magnetization_inclination = inclination
        magnetization_declination = declination
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


def _check_magnetization_angles(magnetization_inclination, magnetization_declination):
    """
    Check if magnetization angles are both None or both numbers.

    They could either be two Nones or two angles, but not one None and one
    angle.
    """
    if magnetization_inclination is None and magnetization_declination is not None:
        raise ValueError(
            "Invalid magnetization degrees. Found `magnetization_inclination` as "
            + "None and `magnetization_declination` as"
            + f"'{magnetization_declination}'. "
            "Please, provide two valid angles in degrees or both angles as None."
        )
    if magnetization_declination is None and magnetization_inclination is not None:
        raise ValueError(
            "Invalid magnetization degrees. Found `magnetization_declination` as "
            + "None and `magnetization_inclination` as"
            + f"'{magnetization_inclination}'. "
            "Please, provide two valid angles in degrees or both angles as None."
        )
