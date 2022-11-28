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
    da_filter = np.sqrt(k_easting ** 2 + k_northing ** 2) ** order
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
    da_filter = np.exp(-np.sqrt(k_easting ** 2 + k_northing ** 2) * height_displacement)
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
        -(k_easting ** 2 + k_northing ** 2) / (2 * (2 * np.pi / wavelength) ** 2)
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
        -(k_easting ** 2 + k_northing ** 2) / (2 * (2 * np.pi / wavelength) ** 2)
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

        g(\mathbf{k}) = \frac{|\mathbf{k}|}{i(\mathbf{ke}\cos{(inclination)}
        \sin{(declination)}+\mathbf{kn}\cos{(inclination)}\cos{(declination)})+
        |\mathbf{k}|\sin{(inclination)}}\times\frac{|\mathbf{k}|}
        {i(\mathbf{ke}\cos{(magnetization\_inclination)}
        \sin{(magnetization\_declination)}+\mathbf{kn}\
        cos{(magnetization\_inclination)}\cos{(magnetization\_declination)})
        +|\mathbf{k}|\sin{(magnetization\_inclination)}}

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}`
    where :math:`\mathbf{f}` is the frequency vector,
    :math:`\mathbf{fe}` is the easting frequency vector,
    :math:`\mathbf{fn}` is the northing frequency vector).

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    inclination : float in degrees
        The inclination inducing Geomagnetic field.
    declination : float in degrees
        The declination inducing Geomagnetic field.
    magnetization_inclination : float in degrees
        The inclination of the total magnetization of the anomaly source.
        Default is i, neglecting remanent magnetization and
        self demagnetization.
    magnetization_declination : float in degrees
        The declination of the total magnetization of the anomaly source.
        Default is d, neglecting remanent magnetization and
        self demagnetization.

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
    # Transform degree to rad
    [inclination, declination] = np.deg2rad([inclination, declination])

    if magnetization_declination is None or magnetization_inclination is None:
        [magnetization_inclination, magnetization_declination] = [
            inclination,
            declination,
        ]
    else:
        [magnetization_inclination, magnetization_declination] = np.deg2rad(
            [magnetization_inclination, magnetization_declination]
        )
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for reduction to pole in frequency domain
    da_filter = (k_northing ** 2 + k_easting ** 2) / (
        (
            1j
            * (
                np.cos(inclination) * np.sin(declination) * k_easting
                + np.cos(inclination) * np.cos(declination) * k_northing
            )
            + np.sin(inclination) * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
        * (
            1j
            * (
                np.cos(magnetization_inclination)
                * np.sin(magnetization_declination)
                * k_easting
                + np.cos(magnetization_inclination)
                * np.cos(magnetization_declination)
                * k_northing
            )
            + np.sin(magnetization_inclination)
            * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
    )

    # Deal with inf and nan value
    da_filter.data = np.nan_to_num(da_filter.data, posinf=1, nan=1)
    return da_filter


def pseudo_gravity_kernel(
    fft_grid,
    inclination=90,
    declination=0,
    magnetization_inclination=None,
    magnetization_declination=None,
    f=50000,
):
    r"""
    Filter for pseudo gravity in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the pseudo gravity. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = \frac{149.8}{f|\mathbf{k}|}\times\frac{|\mathbf{k}|}
        {i(\mathbf{ke}\cos{(inclination)}\sin{(declination)}+
        \mathbf{kn}\cos{(inclination)}\cos{(declination)})+|\mathbf{k}|
        \sin{(inclination)}}\times\frac{|\mathbf{k}|}{i(\mathbf{ke}\
        cos{(magnetization\_inclination)}\sin{(magnetization\_declination)}+
        \mathbf{kn}\cos{(magnetization\_inclination)}
        \cos{(magnetization\_declination)})+|\mathbf{k}|\sin{(magnetization\_inclination)}}


    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector,:math:`\mathbf{fe}` is the easting frequency vector,
    :math:`\mathbf{fn}` is the northing frequency vector).

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Total Magnetic Intensity or Reduction To Pole magnetic field
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    inclination : float in degrees
        The inclination inducing Geomagnetic field. Default is 90 degree for
        RTP field.
    declination : float in degrees
        The declination inducing Geomagnetic field. Default is 0 degree for
        RTP field.
    magnetization_inclination : float in degrees
        The inclination of the total magnetization of the anomaly source.
        Default is i, neglecting remanent magnetization and
        self demagnetization.
    magnetization_declination : float in degrees
        The declination of the total magnetization of the anomaly source.
        Default is d, neglecting remanent magnetization and
        self demagnetization.
    f : float or 2d-array
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
    # Transform degree to rad
    [inclination, declination] = np.deg2rad([inclination, declination])

    if magnetization_declination is None or magnetization_inclination is None:
        [magnetization_inclination, magnetization_declination] = [
            inclination,
            declination,
        ]
    else:
        [magnetization_inclination, magnetization_declination] = np.deg2rad(
            [magnetization_inclination, magnetization_declination]
        )
    # Catch the dims of the Fourier transformed grid
    dims = fft_grid.dims
    # Grab the coordinates of the Fourier transformed grid
    freq_easting = fft_grid.coords[dims[1]]
    freq_northing = fft_grid.coords[dims[0]]
    # Convert frequencies to wavenumbers
    k_easting = 2 * np.pi * freq_easting
    k_northing = 2 * np.pi * freq_northing
    # Compute the filter for reduction to pole in frequency domain
    da_filter = (k_northing ** 2 + k_easting ** 2) / (
        (
            1j
            * (
                np.cos(inclination) * np.sin(declination) * k_easting
                + np.cos(inclination) * np.cos(declination) * k_northing
            )
            + np.sin(inclination) * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
        * (
            1j
            * (
                np.cos(magnetization_inclination)
                * np.sin(magnetization_declination)
                * k_easting
                + np.cos(magnetization_inclination)
                * np.cos(magnetization_declination)
                * k_northing
            )
            + np.sin(magnetization_inclination)
            * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
    )

    # Combine with vertical intergral
    da_filter = da_filter * np.sqrt(k_easting ** 2 + k_northing ** 2) ** -1
    # Deal with inf and nan value
    da_filter.data = np.nan_to_num(da_filter.data, posinf=1, nan=1)

    return da_filter / 149.8 / f
