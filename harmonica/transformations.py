# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Apply transformations to regular grids of potential fields
"""
from .filters._filters import (
    derivative_easting_kernel,
    derivative_northing_kernel,
    derivative_upward_kernel,
    gaussian_highpass_kernel,
    gaussian_lowpass_kernel,
    pseudo_gravity_kernel,
    reduction_to_pole_kernel,
    upward_continuation_kernel,
)
from .filters._utils import apply_filter


def derivative_upward(grid, order=1):
    """
    Calculate the derivative of a potential field grid in the upward direction

    Compute the spatial derivative in the upward direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the upward derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.derivative_upward_kernel
    """
    return apply_filter(grid, derivative_upward_kernel, order=order)


def derivative_easting(grid, order=1):
    """
    Calculate the derivative of a potential field grid in the easting direction

    Compute the spatial derivative in the easting direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the easting derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.derivative_easting_kernel
    """
    return apply_filter(grid, derivative_easting_kernel, order=order)


def derivative_northing(grid, order=1):
    """
    Calculate the derivative of a potential field grid in the northing
    direction

    Compute the spatial derivative in the northing direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the northing derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.derivative_northing_kernel
    """
    return apply_filter(grid, derivative_northing_kernel, order=order)


def upward_continuation(grid, height_displacement):
    """
    Calculate the upward continuation of a potential field grid

    Compute the upward continuation of regular gridded data using frequency
    domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    height_displacement : float
        The height displacement of upward continuation. For upward
        continuation, the height displacement should be positive. Its units
        are the same units of the ``grid`` coordinates.

    Returns
    -------
    upward continuation : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after upward continuation of the passed
        ``grid``.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.upward_continuation_kernel
    """
    return apply_filter(
        grid, upward_continuation_kernel, height_displacement=height_displacement
    )


def gaussian_lowpass(grid, wavelength):
    """
    Calculate the gaussian low-pass of a potential field grid

    Compute the gaussian low-pass of regular gridded data using frequency
    domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    wavelength : float
        The cutoff wavelength in low-pass filter. Its units are the same units
        of the ``grid`` coordinates.

    Returns
    -------
    gaussian lowpass : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after gaussian low-pass of the passed
        ``grid``.

    References
    ----------
    [Geosoft1999]_

    See also
    --------
    harmonica.filters.gaussian_lowpass_kernel
    """
    return apply_filter(grid, gaussian_lowpass_kernel, wavelength=wavelength)


def gaussian_highpass(grid, wavelength):
    """
    Calculate the gaussian high-pass of a potential field grid

    Compute the gaussian high-pass of regular gridded data using frequency
    domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    wavelength : float
        The cutoff wavelength in high-pass filter. Its units are the same
        units of the ``grid`` coordinates.

    Returns
    -------
    gaussian highpass : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after gaussian high-pass of the passed
        ``grid``.

    References
    ----------
    [Geosoft1999]_

    See also
    --------
    harmonica.filters.gaussian_highpass_kernel
    """
    return apply_filter(grid, gaussian_highpass_kernel, wavelength=wavelength)


def reduction_to_pole(
    grid,
    inclination,
    declination,
    magnetization_inclination=None,
    magnetization_declination=None,
):
    """
    Calculate the reduction to the pole of a magnetic field grid

    Compute the reduction to the pole of regular gridded magnetic data using
    frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
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
    RTP : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after reduction to the pole of the passed
        ``grid``.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.reduction_to_pole_kernel
    """
    return apply_filter(
        grid,
        reduction_to_pole_kernel,
        inclination=inclination,
        declination=declination,
        magnetization_inclination=magnetization_inclination,
        magnetization_declination=magnetization_declination,
    )


def pseudo_gravity(
    grid,
    inclination=90,
    declination=0,
    magnetization_inclination=None,
    magnetization_declination=None,
    ambient_field=50000,
):
    """
    Calculate the pseudo gravity of a magnetic field grid

    Compute the pseudo gravity of regular gridded magnetic data using frequency
    domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
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
    ambient_field : float or :class:`xarray.DataArray` in nT
        Ambient field in the study area. It can use the mean ambient field
        value in the study area or the real ambient field value in all
        locations. Default is 50,000 nT.

    Returns
    -------
    pseudo gravity : :class:`xarray.DataArray`
        A pseudo gravity :class:`xarray.DataArray` of the passed``grid``.
        Its units are the same units of the ``grid`` multiply units of its
        coordinates. The vertical integral of the reduction to pole magnetic
        field is normalised by the ambient field. It reflects pseudo gravity of
        a geological body based on cgs susceptibility units.

    References
    ----------
    [Salem2014]_

    See also
    --------
    harmonica.filters.pseudo_gravity_kernel
    """
    return apply_filter(
        grid,
        pseudo_gravity_kernel,
        inclination=inclination,
        declination=declination,
        magnetization_inclination=magnetization_inclination,
        magnetization_declination=magnetization_declination,
        ambient_field=ambient_field,
    )
