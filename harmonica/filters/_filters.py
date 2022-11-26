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

        g(\mathbf{k}) = (\mathbf{ke}*i) ^ n

    where :math:`\mathbf{ke}` is the wavenumber vector
    (:math:`\mathbf{ke} = 2\pi \mathbf{fe}` where :math:`\mathbf{fe}` is the
    easting frequency vector) and :math:`n` is the order of the derivative.

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

        g(\mathbf{k}) = (\mathbf{kn}*i) ^ n

    where :math:`\mathbf{kn}` is the wavenumber vector
    (:math:`\mathbf{kn} = 2\pi \mathbf{fn}` where :math:`\mathbf{fn}` is the
    northing frequency vector) and :math:`n` is the order of the derivative.

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


def upward_continuation_kernel(fft_grid, height):
    r"""
    Filter for upward continuation in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the upward continuation. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = e^{|\mathbf{k}| ^ h}

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and :math:`h` is the height of the upward continuation.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
        Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` functions to
        compute the Fourier Transform and its inverse, respectively.
    height : float
        The height of upward continuation. Value should be negative. It has the
        same units as the input xarray data.

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
    da_filter = np.exp(np.sqrt(k_easting ** 2 + k_northing ** 2) * height)
    return da_filter


def gaussian_lowpass_kernel(fft_grid, wavelength):
    r"""
    Filter for gaussian low-pass in frequency domain

    Return a :class:`xarray.DataArray` with the values of a Gaussian low-pass filter the frequency domain.
    The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = e^\frac{-|\mathbf{k}| ^ 2}{\mathbf{kw}^2}

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and `\mathbf{kw}` is the wavenumber of the cutoff
    wavelength
    :math:`\mathbf{w}` (:math:`\mathbf{kw} = \frac{2\pi} {\mathbf{w}}`).

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
        input xarray data.

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

        g(\mathbf{k}) = 1-e^\frac{-|\mathbf{k}| ^ 2}{(2\mathbf{kw})^2}

    where :math:`\mathbf{k}` is the wavenumber vector
    (:math:`\mathbf{k} = 2\pi \mathbf{f}` where :math:`\mathbf{f}` is the
    frequency vector) and `\mathbf{kw}` is the wavenumber of the cutoff
    wavelength
    :math:`\mathbf{w}` (:math:`\mathbf{kw} = \frac{2\pi} {\mathbf{w}}`).

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
        input xarray data.

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


def reduction_to_pole_kernel(fft_grid, i, d, im=None, dm=None):
    r"""
    Filter for reduction to the pole in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for applying a reduction to the pole on magnetic data. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = \frac{\mathbf{k}}{i(\mathbf{ke}\cos{i}\sin{d}+
        \mathbf{kn}\cos{i}\cos{d})+\mathbf{k}\sin{i}}\times\frac{\mathbf{k}}
        {i(\mathbf{ke}\cos{im}\sin{im}+\mathbf{kn}\cos{im}\cos{dm})+\mathbf{k}
        \sin{im}}

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
    i : float in degrees
        The inclination inducing Geomagnetic field.
    d : float in degrees
        The declination inducing Geomagnetic field.
    im : float in degrees
        The inclination of the total magnetization of the anomaly source.
        Default is i, neglecting remanent magnetization and
        self demagnetization.
    dm : float in degrees
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
    [i, d] = np.deg2rad([i, d])

    if dm is None or im is None:
        [im, dm] = [i, d]
    else:
        [im, dm] = np.deg2rad([im, dm])
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
            * (np.cos(i) * np.sin(d) * k_easting + np.cos(i) * np.cos(d) * k_northing)
            + np.sin(i) * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
        * (
            1j
            * (
                np.cos(im) * np.sin(dm) * k_easting
                + np.cos(im) * np.cos(dm) * k_northing
            )
            + np.sin(im) * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
    )

    # Deal with inf and nan value
    da_filter.data = np.nan_to_num(da_filter.data, posinf=0, nan=0)
    return da_filter


def pseudo_gravity_kernel(fft_grid, i=90, d=0, im=None, dm=None, f=50000):
    r"""
    Filter for pseudo gravity in frequency domain

    Return a :class:`xarray.DataArray` with the values of the frequency domain
    filter for computing the upward derivative. The filter is built upon the
    frequency coordinates of the passed ``fft_grid`` and is defined as follows:

    .. math::

        g(\mathbf{k}) = |\mathbf{k}| ^ {-1}\times\frac{|\mathbf{k}|}
        {i(\mathbf{ke}\cos{i}\sin{d}+\mathbf{kn}\cos{i}\cos{d})+\mathbf{k}\sin{i}}
        \times\frac{|\mathbf{k}|}{i(\mathbf{ke}\cos{im}\sin{dm}+
        \mathbf{kn}\cos{im}\cos{dm})+\mathbf{k}\sin{im}}

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
    i : float in degrees
        The inclination inducing Geomagnetic field. Default is 90 degree for
        RTP field.
    d : float in degrees
        The declination inducing Geomagnetic field. Default is 0 degree for
        RTP field.
    im : float in degrees
        The inclination of the total magnetization of the anomaly source.
        Default is i, neglecting remanent magnetization and
        self demagnetization.
    dm : float in degrees
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
    [Blakely1995]_

    See also
    --------
    harmonica.pseudo_gravity
    """
    # Transform degree to rad

    [i, d] = np.deg2rad([i, d])

    if dm is None or im is None:
        [im, dm] = [i, d]
    else:
        [im, dm] = np.deg2rad([im, dm])
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
            * (np.cos(i) * np.sin(d) * k_easting + np.cos(i) * np.cos(d) * k_northing)
            + np.sin(i) * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
        * (
            1j
            * (
                np.cos(im) * np.sin(dm) * k_easting
                + np.cos(im) * np.cos(dm) * k_northing
            )
            + np.sin(im) * np.sqrt(k_northing ** 2 + k_easting ** 2)
        )
    )

    # Combine with vertical intergral
    da_filter = da_filter * np.sqrt(k_easting ** 2 + k_northing ** 2) ** -1
    # Deal with inf and nan value
    da_filter.data = np.nan_to_num(da_filter.data, posinf=0, nan=0)

    return da_filter / 149.8 / f
