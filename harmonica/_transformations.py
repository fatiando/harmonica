# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Apply transformations to regular grids of potential fields
"""
from .filters import derivative_upward_kernel
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
