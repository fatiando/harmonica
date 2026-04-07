# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Apply transformations to regular grids of potential fields.
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
from .filters._utils import apply_filter, grid_sanity_checks


def derivative_upward(grid, *, order=1, pad=True, pad_kwargs=None):
    """
    Calculate the derivative of a potential field grid in the upward direction.

    Compute the spatial derivative in the upward direction of regular gridded
    data using frequency domain calculations through Fast Fourier Transform.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    order : int, optional
        The order of the derivative. Default to 1.
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the upward derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.filters.derivative_upward_kernel
    """
    return apply_filter(
        grid,
        derivative_upward_kernel,
        filter_kwargs={"order": order},
        pad=pad,
        pad_kwargs=pad_kwargs,
    )


def derivative_easting(
    grid, *, order=1, method="finite-diff", pad=True, pad_kwargs=None
):
    """
    Calculate the derivative of a regular grid in the easting direction.

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
    order : int, optional
        The order of the derivative. Default to 1.
    method : str, optional
        Method that will be used for computing the easting derivative. It can
        be either ``"finite-diff"``, for computing using
        :func:`xarray.differentiate`, or ``"fft"``, for using FFT-based
        filters. Default ``"finite-diff"``.
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the easting derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates to the power of the passed ``order``.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.filters.derivative_easting_kernel
    """
    if method == "finite-diff":
        coordinate = _get_dataarray_coordinate(grid, dimension_index=1)
        for _ in range(order):
            grid = grid.differentiate(coord=coordinate)
    elif method == "fft":
        grid = apply_filter(
            grid,
            derivative_easting_kernel,
            filter_kwargs={"order": order},
            pad=pad,
            pad_kwargs=pad_kwargs,
        )
    else:
        msg = (
            f"Invalid method '{method}'. Please select one from 'finite-diff' or 'fft'."
        )
        raise ValueError(msg)
    return grid


def derivative_northing(
    grid, *, order=1, method="finite-diff", pad=True, pad_kwargs=None
):
    """
    Calculate the derivative of a regular grid in the northing direction.

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
    order : int, optional
        The order of the derivative. Default to 1.
    method : str, optional
        Method that will be used for computing the easting derivative. It can
        be either ``"finite-diff"``, for computing using
        :func:`xarray.differentiate`, or ``"fft"``, for using FFT-based
        filters. Default ``"finite-diff"``.
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the northing derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates to the power of the passed ``order``.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.filters.derivative_northing_kernel
    """
    if method == "finite-diff":
        coordinate = _get_dataarray_coordinate(grid, dimension_index=0)
        for _ in range(order):
            grid = grid.differentiate(coord=coordinate)
    elif method == "fft":
        return apply_filter(
            grid,
            derivative_northing_kernel,
            filter_kwargs={"order": order},
            pad=pad,
            pad_kwargs=pad_kwargs,
        )
    else:
        msg = (
            f"Invalid method '{method}'. Please select one from 'finite-diff' or 'fft'."
        )
        raise ValueError(msg)
    return grid


def upward_continuation(grid, height_displacement, *, pad=True, pad_kwargs=None):
    """
    Calculate the upward continuation of a potential field grid.

    Compute the upward continuation of regular gridded data using frequency
    domain calculations through Fast Fourier Transform.

    .. note::

        Any non-dimensional coordinates of the grid will be dropped since
        upward continuation may have made them no longer correct.

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
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    upward continuation : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after upward continuation of the passed
        ``grid``.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.filters.upward_continuation_kernel
    """
    return apply_filter(
        grid,
        upward_continuation_kernel,
        filter_kwargs={"height_displacement": height_displacement},
        pad=pad,
        pad_kwargs=pad_kwargs,
        drop_coords=True,
    )


def gaussian_lowpass(grid, wavelength, *, pad=True, pad_kwargs=None):
    """
    Calculate the Gaussian low-pass of a potential field grid.

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
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    gaussian lowpass : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after Gaussian low-pass of the passed
        ``grid``.

    References
    ----------
    [Geosoft1999]_

    See Also
    --------
    harmonica.filters.gaussian_lowpass_kernel
    """
    return apply_filter(
        grid,
        gaussian_lowpass_kernel,
        pad=pad,
        pad_kwargs=pad_kwargs,
        filter_kwargs={"wavelength": wavelength},
    )


def gaussian_highpass(grid, wavelength, *, pad=True, pad_kwargs=None):
    """
    Calculate the Gaussian high-pass of a potential field grid.

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
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    gaussian highpass : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after Gaussian high-pass of the passed
        ``grid``.

    References
    ----------
    [Geosoft1999]_

    See Also
    --------
    harmonica.filters.gaussian_highpass_kernel
    """
    return apply_filter(
        grid,
        gaussian_highpass_kernel,
        pad=pad,
        pad_kwargs=pad_kwargs,
        filter_kwargs={"wavelength": wavelength},
    )


def reduction_to_pole(
    grid,
    inclination,
    declination,
    magnetization_inclination=None,
    magnetization_declination=None,
    *,
    pad=True,
    pad_kwargs=None,
):
    """
    Calculate the reduction to the pole of a magnetic field grid.

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
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    reduced_to_pole_grid : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after reduction to the pole of the passed
        ``grid``.

    References
    ----------
    [Blakely1995]_

    See Also
    --------
    harmonica.filters.reduction_to_pole_kernel
    """
    return apply_filter(
        grid,
        reduction_to_pole_kernel,
        filter_kwargs={
            "inclination": inclination,
            "declination": declination,
            "magnetization_inclination": magnetization_inclination,
            "magnetization_declination": magnetization_declination,
        },
        pad=pad,
        pad_kwargs=pad_kwargs,
    )


def total_gradient_amplitude(grid, *, pad=True, pad_kwargs=None):
    r"""
    Calculate the total gradient amplitude of a potential field grid.

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
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    total_gradient_amplitude_grid : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` after calculating the total gradient
        amplitude of the passed ``grid``.

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
    # Run sanity checks on the grid
    grid_sanity_checks(grid)
    # Calculate the gradients of the grid
    gradient = (
        derivative_easting(grid, order=1, pad=pad, pad_kwargs=pad_kwargs),
        derivative_northing(grid, order=1, pad=pad, pad_kwargs=pad_kwargs),
        derivative_upward(grid, order=1, pad=pad, pad_kwargs=pad_kwargs),
    )
    # return the total gradient amplitude
    return np.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2)


def tilt_angle(grid, *, pad=True, pad_kwargs=None):
    r"""
    Calculate the tilt angle of a potential field grid.

    Compute the tilt of a regular gridded potential field :math:`M`. The
    horizontal derivatives are calculated through finite-differences while the
    upward derivative is calculated using FFT.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    pad : bool, optional
        If True, will add padding to the grid before taking the Fourier Transform
        and applying the filter and remove it after the inverse Fourier Transform.
        Adding padding usually helps reduce edge effects from signal truncation.
        Default is True.
    pad_kwargs : dict or None, optional
        Any additional keyword arguments that should be passed to the
        :meth:`xarray.DataArray.pad` function in the form of a dictionary. If none
        are given, the default padding of 25% the dimensions of the grid will be
        added using the "edge" method.

    Returns
    -------
    tilt_grid : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the calculated tilt
        in radians.

    Notes
    -----
    The tilt is calculated as:

    .. math::

        \text{tilt}(f) = \tan^{-1} \left(
            \frac{
                \frac{\partial M}{\partial z}
            }{
                \sqrt{
                    \left( \frac{\partial M}{\partial x} \right)^2
                    +
                    \left( \frac{\partial M}{\partial y} \right)^2
                }
            }
        \right)

    where :math:`M` is the regularly gridded potential field.

    References
    ----------
    [Blakely1995]_
    [MillerSingh1994]_
    """
    grid_sanity_checks(grid)
    deriv_east = derivative_easting(grid, order=1, pad=pad, pad_kwargs=pad_kwargs)
    deriv_north = derivative_northing(grid, order=1, pad=pad, pad_kwargs=pad_kwargs)
    deriv_up = derivative_upward(grid, order=1, pad=pad, pad_kwargs=pad_kwargs)
    horiz_deriv = np.hypot(deriv_east, deriv_north)
    tilt = np.arctan2(deriv_up, horiz_deriv)
    return tilt


def _get_dataarray_coordinate(grid, dimension_index):
    """
    Return the name of the easting or northing coordinate in the grid.

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
        direction = "northing" if dimension_index == 0 else "easting"
        coords = "', '".join(coords)
        msg = (
            f"Grid contains more than one coordinate along the '{direction}' "
            f"direction: '{coords}'."
        )
        raise ValueError(msg)
    return coords[0]
