# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Custom FFT and inverse FFT functions that work with :class:`xarray.DataArray`.

These functions are inspired in the ``fft`` and ``ifft`` functions provided by ``xrft``,
which are released under the MIT license.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr


def fft(grid, *, prefix="freq_"):
    """
    Compute Fast Fourier Transform of a 2D regular grid.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A two dimensional :class:`xarray.DataArray` whose coordinates are
        evenly spaced (regular grid). Its dimensions should be in the following
        order: *northing*, *easting*. Its coordinates should be defined in the
        same units.
    prefix : str, optional
        Prefix used for the name of the frequency coordinates and dimensions.


    Returns
    -------
    fourier_transform : :class:`xarray.DataArray`
        Array with the Fourier transform of the original grid.
    """
    if not isinstance(grid, xr.DataArray):
        msg = (
            f"Invalid 'grid' of type '{type(grid).__name__}'. "
            "It must be an xarray.DataArray."
        )
        raise TypeError(msg)
    if grid.ndim != 2:
        msg = (
            f"Invalid grid array with '{grid.ndim}' dimensions. It must be a 2D array."
        )
        raise ValueError(msg)

    # Get dimensional coordinates and coordinates' shifts
    dimensional_coords = tuple(
        _get_dimensional_coordinate(grid, dim) for dim in grid.dims
    )
    shifts = tuple(grid.coords[coord].values.min() for coord in dimensional_coords)

    # Generate new coordinates
    fft_dims = tuple(f"{prefix}{dim}" for dim in grid.dims)
    fft_coords = {
        f"{prefix}{coord}": (dim, _fftfreq(grid.coords[coord]))
        for coord, dim in zip(dimensional_coords, fft_dims, strict=True)
    }

    # Compute FFT
    fft = np.fft.fftshift(np.fft.fftn(grid.values))

    # Build new xr.DataArray
    da_fft = xr.DataArray(fft, dims=fft_dims, coords=fft_coords)

    # Add shifts to frequency coordinates
    for coord, shift in zip(fft_coords, shifts, strict=True):
        da_fft.coords[coord].attrs.update({"shift": shift})
    return da_fft


def ifft(fft_grid, *, prefix="freq_"):
    """
    Compute the inverse Fast Fourier Transform of a 2D regular grid.

    If the frequency coordinates have a *shift* attribute, it will be used to shift the
    coordinates in the spatial domain to such value.

    .. important::

        Assumes that the ``fft_grid`` is *shifted*: it was passed to
        :func:`numpy.fft.fftshift`. The outputs of the ``fft`` function satisfy this
        condition.

    Parameters
    ----------
    fft_grid : :class:`xarray.DataArray`
        Array with a regular grid defined in the frequency domain.
        Its dimensions should be in the following order:
        *freq_northing*, *freq_easting*.
    prefix : str, optional
        Prefix used for the name of the frequency coordinates and dimensions.

    Returns
    -------
    grid : :class:`xarray.DataArray`
        Array with the inverse Fourier transform of the passed grid.
    """
    if not isinstance(fft_grid, xr.DataArray):
        msg = (
            f"Invalid 'grid' of type '{type(fft_grid).__name__}'. "
            "It must be an xarray.DataArray."
        )
        raise TypeError(msg)
    if fft_grid.ndim != 2:
        msg = (
            f"Invalid grid array with '{fft_grid.ndim}' dimensions. "
            "It must be a 2D array."
        )
        raise ValueError(msg)

    for dim in fft_grid.dims:
        if not dim.startswith(prefix):
            msg = (
                f"Invalid frequency dimension '{dim}'. "
                f"It doesn't start with prefix '{prefix}'."
            )
            raise ValueError(msg)

    # Get dimensional frequency coordinates and spacings
    dimensional_fft_coords = tuple(
        _get_dimensional_coordinate(fft_grid, dim) for dim in fft_grid.dims
    )
    for coord in dimensional_fft_coords:
        if not coord.startswith(prefix):
            msg = (
                f"Invalid dimensional coordinate '{coord}'. "
                f"It doesn't start with prefix '{prefix}'."
            )
            raise ValueError(msg)

    # Generate new coordinates
    dims = tuple(dim.removeprefix(prefix) for dim in fft_grid.dims)

    coords = {
        coord.removeprefix(prefix): (dim, _ifftfreq(fft_grid.coords[coord]))
        for coord, dim in zip(dimensional_fft_coords, dims, strict=True)
    }

    # Compute iFFT
    ifft = np.fft.ifftn(np.fft.ifftshift(fft_grid.values))

    # Build new xr.DataArray
    da = xr.DataArray(ifft, dims=dims, coords=coords)

    return da


def _get_spacing(coordinate: xr.DataArray) -> float:
    """
    Return spacing of a grid coordinate.

    Parameters
    ----------
    coordinate : xarray.DataArray
        DataArray containing the coordinate.
    coordinate : str
        Coordinate name.

    Returns
    -------
    spacing : float
    """
    spacing = coordinate.values[1] - coordinate.values[0]
    if not np.allclose(spacing, coordinate.values[1:] - coordinate.values[:-1]):
        msg = f"Invalid '{coordinate.name}' coordinates: they must be evenly spaced."
        raise ValueError(msg)
    if spacing <= 0:
        msg = (
            f"Invalid coordinate '{coordinate.name}': it must be increasingly ordered."
        )
        raise ValueError(msg)
    return spacing


def _get_dimensional_coordinate(grid: xr.DataArray, dim: str) -> str:
    """
    Get dimensional coordinate in the grid for a particular dimension.

    Parameters
    ----------
    grid : xarray.DataArray
        DataArray containing the coordinate.
    dim : str
        Dimension name.

    Returns
    -------
    dimensional_coordinate : str
    """
    potential_coords = [
        coord for coord in grid.coords if grid.coords[coord].dims == (dim,)
    ]
    if not potential_coords:
        msg = f"Couldn't find dimensional coordinate for dimension '{dim}'."
        raise ValueError(msg)
    if len(potential_coords) > 1:
        bad_coords = ", ".join(potential_coords)
        msg = (
            f"Multiple dimensional coordinates ({bad_coords}) found "
            f"for the '{dim}' dimension. "
            "Leave only one dimensional coordinate per dimension."
        )
        raise ValueError(msg)
    (dimensional_coordinate,) = potential_coords
    return dimensional_coordinate


def _fftfreq(coordinate: xr.DataArray) -> npt.NDArray:
    """
    Get coordinate into the frequency domain.
    """
    if coordinate.ndim != 1:
        raise ValueError()
    spacing = _get_spacing(coordinate)
    return np.fft.fftshift(np.fft.fftfreq(coordinate.size, spacing))


def _ifftfreq(freq: xr.DataArray) -> npt.NDArray:
    """
    Recover coordinate in the space domain from the frequency domain.

    Shifts the coordinates in the spatial domain if the ``freq`` has a *shift*
    attribute.
    """
    if freq.ndim != 1:
        raise ValueError()
    spacing = _get_spacing(freq)
    coordinate = np.fft.fftshift(np.fft.fftfreq(freq.size, spacing))

    # Apply static shift if any
    if "shift" in freq.attrs:
        coordinate += freq.attrs["shift"] - coordinate.min()

    return coordinate
