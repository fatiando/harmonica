# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Apply transformations to regular grids of potential fields
"""
import numpy as np

from .filters._filters import (
    derivative_easting_kernel,
    derivative_northing_kernel,
    derivative_upward_kernel,
    gaussian_highpass_kernel,
    gaussian_lowpass_kernel,
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


def derivative_easting(grid, order=1, method="finite-diff"):
    """
    Calculate the derivative of a regular grid in the easting direction

    Compute the spatial derivative in the easting direction of regular gridded
    data. It can compute using accurate central differences using
    :func:`xarray.differentiate` or through frequency domain calculations
    through Fast Fourier Transform.

    .. important::

        Choosing the finite differences option produces more accurate results
        without border effects.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.
    method : str (optional)
        Method that will be used for computing the easting derivative. It can
        be either ``"finite-diff"``, for computing using
        :func:`xarray.differentiate`, or ``"fft"``, for using FFT-based
        filters.
        Default ``"finite-diff"``.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the easting derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates to the power of the passed ``order``.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.derivative_easting_kernel
    """
    if method == "finite-diff":
        coordinate = _get_dataarray_coordinate(grid, dimension_index=1)
        for _ in range(order):
            grid = grid.differentiate(coord=coordinate)
    elif method == "fft":
        grid = apply_filter(grid, derivative_easting_kernel, order=order)
    else:
        raise ValueError(
            f"Invalid method '{method}'. "
            "Please select one from 'finite-diff' or 'fft'."
        )
    return grid


def derivative_northing(grid, order=1, method="finite-diff"):
    """
    Calculate the derivative of a regular grid in the northing direction

    Compute the spatial derivative in the northing direction of regular gridded
    data. It can compute using accurate central differences using
    :func:`xarray.differentiate` or through frequency domain calculations
    through Fast Fourier Transform.

    .. important::

        Choosing the finite differences option produces more accurate results
        without border effects.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int
        The order of the derivative. Default to 1.
    method : str (optional)
        Method that will be used for computing the easting derivative. It can
        be either ``"finite-diff"``, for computing using
        :func:`xarray.differentiate`, or ``"fft"``, for using FFT-based
        filters.
        Default ``"finite-diff"``.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the northing derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates to the power of the passed ``order``.

    References
    ----------
    [Blakely1995]_

    See also
    --------
    harmonica.filters.derivative_northing_kernel
    """
    if method == "finite-diff":
        coordinate = _get_dataarray_coordinate(grid, dimension_index=0)
        for _ in range(order):
            grid = grid.differentiate(coord=coordinate)
    elif method == "fft":
        return apply_filter(grid, derivative_northing_kernel, order=order)
    else:
        raise ValueError(
            f"Invalid method '{method}'. "
            "Please select one from 'finite-diff' or 'fft'."
        )
    return grid


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
    Calculate the Gaussian low-pass of a potential field grid

    Compute the Gaussian low-pass of regular gridded data using frequency
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
        A :class:`xarray.DataArray` after Gaussian low-pass of the passed
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
    Calculate the Gaussian high-pass of a potential field grid

    Compute the Gaussian high-pass of regular gridded data using frequency
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
        A :class:`xarray.DataArray` after Gaussian high-pass of the passed
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
    reduced_to_pole_grid : :class:`xarray.DataArray`
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


def total_gradient_amplitude(grid):
    r"""
    Calculate the total gradient amplitude a magnetic field grid

    Compute the total gradient amplitude of a regular gridded potential field
    `M`. The horizontal derivatives are calculated though finite-differences
    while the upward derivative is calculated using FFT.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.

    Returns
    -------
    total_gradient_amplitude_grid : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after calculating the
        total gradient amplitude of the passed ``grid``.

    Notes
    -----
    The total gradient amplitude is calculated as:

    .. math::

        A(x, y) = \sqrt{
            \left( \frac{\partial M}{\partial x} \right)^2
            + \left( \frac{\partial M}{\partial y} \right)^2
            + \left( \frac{\partial M}{\partial z} \right)^2
        }

    where :math:`M` is the regularly gridded potential field.

    References
    ----------
    [Blakely1995]_
    """

    # Catch the dims of the grid
    dims = grid.dims
    # Check if the array has two dimensions
    if len(dims) != 2:
        raise ValueError(
            f"Invalid grid with {len(dims)} dimensions. "
            + "The passed grid must be a 2 dimensional array."
        )
    # Check if the grid has nans
    if np.isnan(grid).any():
        raise ValueError(
            "Found nan(s) on the passed grid. "
            + "The grid must not have missing values before computing the "
            + "Fast Fourier Transform."
        )
    # Calculate the gradients of the grid
    gradient = (
        derivative_easting(grid, order=1),
        derivative_northing(grid, order=1),
        derivative_upward(grid, order=1),
    )
    # return the total gradient amplitude
    return np.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2)


def _get_dataarray_coordinate(grid, dimension_index):
    """
    Return the name of the easting or northing coordinate in the grid

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        Regular grid
    dimension_index : int
        Index of the dimension for the desired coordinate of the regular grid.
        Since the dimensions of the grid should be in the order of
        *northing*, *easting*, then 0 corresponds to *northing* and 1 to
        *easting*.
    """
    dim_name = grid.dims[dimension_index]
    coords = [c for c in grid.coords if grid[c].dims == (dim_name,)]
    if len(coords) > 1:
        if dimension_index == 0:
            direction = "northing"
        else:
            direction = "easting"
        coords = "', '".join(coords)
        raise ValueError(
            f"Grid contains more than one coordinate along the '{direction}' "
            f"direction: '{coords}'."
        )
    return coords[0]
