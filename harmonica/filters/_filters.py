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

from ..constants import GRAVITATIONAL_CONST


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


def derivative_easting_kernel(fft_grid, order=1):
    r"""
    Filter for easting derivative in frequency domain

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

    See also
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
    Filter for northing derivative in frequency domain

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

    See also
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
    Filter for upward continuation in frequency domain

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

    See also
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
    Filter for gaussian low-pass in frequency domain

    Return a :class:`xarray.DataArray` with the values of a Gaussian low-pass
    filter the frequency domain.
    The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

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
        The cutoff wavelength in low-pass filter. It has the same units as the
        input xarray data coordinates.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the gaussian low-pass filter in frequency
        domain.

    References
    ----------
    [Geosoft1999]_

    See also
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
    Filter for gaussian high-pass in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the northing derivative. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

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
        The cutoff wavelength in high-pass filter. It has the same units as the
        input xarray data coordinates.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the Gaussian high-pass filter in frequency
        domain.

    References
    ----------
    [Geosoft1999]_

    See also
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
    Filter for reduction to the pole in frequency domain

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

    where :math:`\hat{\mathbf{f}} = (f_e, f_n, f_z)` is a unit vector parallel
    to the geomagnetic field and :math:`\hat{\mathbf{m}} = (m_e, m_n, m_z)`
    is a unit vector parallel to the magnetization vector of the source. The
    :math:`f_e`, :math:`f_n`, :math:`m_e`, :math:`m_n` are the easting and
    northing components while the :math:`f_z` and :math:`m_z` are the
    **downward** coordinates.
    Each of these components can be obtained from the inclination and
    declination angles of the geomagnetic field (:math:`I` and :math:`D`,
    respectively) and for the magnetization vector (:math:`I_m` and
    :math:`D_m`, respectively):

    .. math::

        \begin{cases}
            f_e = \sin D \cos I \\
            f_n = \cos D \cos I \\
            f_u = \sin I
        \end{cases}

    .. math::

        \begin{cases}
            m_e = \sin D_m \cos I_m \\
            m_n = \cos D_m \cos I_m \\
            m_u = \sin I_m
        \end{cases}

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

    See also
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
    # Compute the filter for reduction to pole in frequency domain
    da_filter = _get_rtp_filter(
        k_easting,
        k_northing,
        inclination,
        declination,
        magnetization_inclination,
        magnetization_declination,
    )
    # Set 0 wavenumber to 0
    da_filter.loc[dict(freq_northing=0, freq_easting=0)] = 0
    return da_filter


def pseudo_gravity_kernel(
    fft_grid,
    inclination=90,
    declination=0,
    magnetization_inclination=None,
    magnetization_declination=None,
    ambient_field=50000,
):
    r"""
    Filter for pseudo gravity in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the pseudo gravity. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

            g(\mathbf{k}) = \frac{gravitational\_constant\times1e^8}
            {ambient\_field\times4\pi}
            \times\frac{reduction\_to\_pole\_kernel}{|\mathbf{k}|}


    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}`

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Total Magnetic Intensity or Reduction To Pole magnetic field.
        Please set inclination = 90 for Reduction To Pole magnetic field.
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    inclination : float in degrees
        The inclination of the inducing Geomagnetic field. Default is 90 degree
         for RTP field.
    declination : float in degrees
        The declination of the inducing Geomagnetic field. Default is 0 degree
         for RTP field.
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
    ambient_field : float or 2d-array
        Ambient field in the study area. It can use the mean ambinent field
        value in the study area or the real ambient field value in all
        locations. Default is 50,000 nT.

    Returns
    -------
    da_filter : :class:`xarray.DataArray`
        Array with the kernel for the pseudo gravity filter in frequency
        domain.

    References
    ----------
    [Salem2014]_

    See also
    --------
    harmonica.pseudo_gravity
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
    # Calculate vertical integral
    da_filter = 1 / np.sqrt(k_easting**2 + k_northing**2)
    # Check if input is RTP field
    if inclination != 90:
        # Add RTP kernel to the calculate vertical integral
        da_filter = (
            _get_rtp_filter(
                k_easting,
                k_northing,
                inclination,
                declination,
                magnetization_inclination,
                magnetization_declination,
            )
            * da_filter
        )
    # Set 0 wavenumber to 0
    da_filter.loc[dict(freq_northing=0, freq_easting=0)] = 0
    return da_filter * GRAVITATIONAL_CONST * 1e8 / ambient_field / 4 / np.pi


def _check_magnetization_angles(magnetization_inclination, magnetization_declination):
    """
    Check if magnetization angles are both None or both numbers

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


def _get_rtp_filter(
    k_easting,
    k_northing,
    inclination,
    declination,
    magnetization_inclination,
    magnetization_declination,
):
    """
    Build the reduction to the pole filter

    Parameters
    ----------
    k_easting : array
        Wavenumber array for the easting direction.
    k_northing : array
        Wavenumber array for the northing direction.
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
    rtp_filter: array
        Array with the kernel for the reduction to the pole filter in frequency
        domain.
    """
    # Convert angles to radians
    inc_rad, dec_rad = np.deg2rad(inclination), np.deg2rad(declination)
    mag_inc_rad, mag_dec_rad = np.deg2rad(magnetization_inclination), np.deg2rad(
        magnetization_declination
    )
    # Compute unit vector components for geomagnetic field and magnetization
    cos_inc, sin_inc = np.cos(inc_rad), np.sin(inc_rad)
    cos_dec, sin_dec = np.cos(dec_rad), np.sin(dec_rad)
    cos_mag_inc, sin_mag_inc = np.cos(mag_inc_rad), np.sin(mag_inc_rad)
    cos_mag_dec, sin_mag_dec = np.cos(mag_dec_rad), np.sin(mag_dec_rad)
    f_e = sin_dec * cos_inc
    f_n = cos_dec * cos_inc
    f_z = sin_inc
    m_e = sin_mag_dec * cos_mag_inc
    m_n = cos_mag_dec * cos_mag_inc
    m_z = sin_mag_inc
    # Precompute |k|^2 and |k|
    k_squared = k_northing**2 + k_easting**2
    k = np.sqrt(k_squared)
    # Compute the rtp filter
    rtp_filter = (
        k_squared
        * (f_z * k + 1j * (f_e * k_easting + f_n * k_northing)) ** (-1)
        * (m_z * k + 1j * (m_e * k_easting + m_n * k_northing)) ** (-1)
    )
    return rtp_filter
