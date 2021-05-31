"""
Compute spatial derivatives of grids in the frequency domain
"""
import numpy as np

from xrft.xrft import fft, ifft


def derivative_easting(grid, order=1, keep_original_coords=True):
    """
    Calculate the derivative of a potential field in the easting direction

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
    keep_original_coords : bool
        If True, the returned :class:`xarray.DataArray` will have the same
        coordinates as the original ``grid``. If False, the coordinates of the
        output :class:`xarray.DataArray` will be the ones returned by
        :func:`xrft.xrft.fft`: centered around zero. Default to True.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the easting derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.
    """
    return _dispatcher(grid, _kernel_easting, order, keep_original_coords)


def derivative_northing(grid, order=1, keep_original_coords=True):
    """
    Calculate the derivative of a potential field in the northing direction

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
    keep_original_coords : bool
        If True, the returned :class:`xarray.DataArray` will have the same
        coordinates as the original ``grid``. If False, the coordinates of the
        output :class:`xarray.DataArray` will be the ones returned by
        :func:`xrft.xrft.fft`: centered around zero. Default to True.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the northing derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.
    """
    return _dispatcher(grid, _kernel_northing, order, keep_original_coords)


def derivative_upward(grid, order=1, keep_original_coords=True):
    """
    Calculate the derivative of a potential field in the upward direction

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
    keep_original_coords : bool
        If True, the returned :class:`xarray.DataArray` will have the same
        coordinates as the original ``grid``. If False, the coordinates of the
        output :class:`xarray.DataArray` will be the ones returned by
        :func:`xrft.xrft.fft`: centered around zero. Default to True.

    Returns
    -------
    derivative : :class:`xarray.DataArray`
        A :class:`xarray.DataArray` with the upward derivatives of the passed
        ``grid``. Its units are the same units of the ``grid`` per units of its
        coordinates.
    """
    return _dispatcher(grid, _kernel_upward, order, keep_original_coords)


def _dispatcher(grid, kernel, order, keep_original_coords):
    """
    Calculate the derivative of a potential field in a particular direction

    Use the passed kernel to obtain the derivative along the chosen direction
    (easting, northing or upward).
    """
    # Catch the dims of the grid
    dims = grid.dims
    # Check if the array has two dimensions
    if len(dims) != 2:
        raise ValueError(
            f"Invalid grid with {len(dims)} dimensions. "
            + "The passed grid must be a 2 dimensional array."
        )
    # Check if grid and coordinates has nans
    if np.isnan(grid).any():
        raise ValueError(
            "Found nan(s) on the passed grid. "
            + "The grid must not have missing values before computing the "
            + "Fast Fourier Transform."
        )
    # Compute Fourier Transform of the grid
    fourier_transform = fft(grid)
    # Compute the derivative in the frequency domain
    deriv_ft = kernel(fourier_transform, order)
    # Compute inverse FFT
    deriv = ifft(deriv_ft).real
    # Move recovered coordinates to original range
    if keep_original_coords:
        coords = {}
        for dim in dims:
            shift = grid.coords[dim].values.min() - deriv.coords[dim].values.min()
            coords[dim] = deriv.coords[dim].values + shift
        deriv = deriv.assign_coords(coords)
    return deriv


def _kernel_easting(fourier_transform, order):
    """
    Compute the easting derivative in the frequency domain

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    deriv_ft : :class:`xarray.DataArray`
        Array with the easting derivative of the original grid in the
        frequency domain.
    """
    # Catch the dims of the Fourier transoformed grid
    dims = fourier_transform.dims
    # Grab the easting coordinates of the Fourier transformed grid
    k_easting = fourier_transform.coords[dims[1]]
    # Compute the easting derivative in the frequency domain
    deriv_ft = fourier_transform * (k_easting * 1j) ** order
    return deriv_ft


def _kernel_northing(fourier_transform, order):
    """
    Compute the northing derivative in the frequency domain

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    deriv_ft : :class:`xarray.DataArray`
        Array with the northing derivative of the original grid in the
        frequency domain.
    """
    # Catch the dims of the Fourier transoformed grid
    dims = fourier_transform.dims
    # Grab the northing coordinates of the Fourier transformed grid
    k_northing = fourier_transform.coords[dims[0]]
    # Compute the northing derivative in the frequency domain
    deriv_ft = fourier_transform * (k_northing * 1j) ** order
    return deriv_ft


def _kernel_upward(fourier_transform, order):
    """
    Compute the upward derivative in the frequency domain

    Parameters
    ----------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
    order : int
        The order of the derivative. Default to 1.

    Returns
    -------
    deriv_ft : :class:`xarray.DataArray`
        Array with the upward derivative of the original grid in the frequency
        domain.
    """
    # Catch the dims of the Fourier transoformed grid
    dims = fourier_transform.dims
    # Grab the coordinates of the Fourier transformed grid
    k_easting = fourier_transform.coords[dims[1]]
    k_northing = fourier_transform.coords[dims[0]]
    # Compute the upward derivative in the frequency domain
    deriv_ft = fourier_transform * np.sqrt(k_easting ** 2 + k_northing ** 2) ** order
    return deriv_ft
