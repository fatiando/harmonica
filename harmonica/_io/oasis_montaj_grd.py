# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Function to read Oasis Montaj© .gdf file
"""

import array

import numpy as np
import xarray as xr


def load_oasis_montaj_grid(fname):
    """
    Reads gridded data from an Oasis Montaj© .grd file.

    The version 2 of the Geosoft© Grid File Format (GRD) stores gridded
    products in binary data. This function can read those files and parse the
    information in the header. It returns the data in
    a :class:`xarray.DataArray` for convenience.

    .. warning::

        This function has not been tested against a wide range of GRD files.
        This could lead to incorrect readings of the stored data. Please report
        any unwanted behaviour by opening an issue in Harmonica:
        https://github.com/fatiando/harmonica/issues

    .. important::

        This function only supports reading GRD files using the **version 2**
        of the Geosoft© Grid File Format.

    .. important::

        This function is not supporting compressed GRD files, rotated grids,
        orderings different than ±1, or colour grids.

    Parameters
    ----------
    fname : (string or file-like object)
        Path to the .grd file.

    Returns
    -------
    grid : :class:`xarray.DataArray`
        :class:`xarray.DataArray` containing the grid, its coordinates and
        header information.

    References
    ----------
    https://help.seequent.com/Oasis-montaj/9.9/en/Content/ss/glossary/grid_file_format__grd.htm
    """
    # Read the header and the grid array
    with open(fname, "rb") as grd_file:
        # Read the header (first 512 bytes)
        header = _read_header(grd_file.read(512))
        # Check for valid flags
        _check_ordering(header["ordering"])
        _check_rotation(header["rotation"])
        _check_sign_flag(header["sign_flag"])
        _check_n_bytes_per_element(header["n_bytes_per_element"])
        # Get data type for the grid elements
        data_type = _get_data_type(header["n_bytes_per_element"], header["sign_flag"])
        # Read grid
        grid = array.array(data_type, grd_file.read())
    # Convert to numpy array
    grid = np.array(grid)
    # Remove dummy values
    grid = _remove_dummies(grid, data_type)
    # Scale the grid
    grid = np.array(grid / header["data_factor"] + header["base_value"])
    # Reshape the grid based on the ordering
    if header["ordering"] == 1:
        order = "C"
        shape = (header["shape_v"], header["shape_e"])
        spacing = (header["spacing_v"], header["spacing_e"])
    elif header["ordering"] == -1:
        order = "F"
        shape = (header["shape_e"], header["shape_v"])
        spacing = (header["spacing_e"], header["spacing_v"])
    grid = grid.reshape(shape, order=order)
    # Build coords
    easting, northing = _build_coordinates(
        header["x_origin"], header["y_origin"], shape, spacing
    )
    # Build an xarray.DataArray for the grid
    dims = ("northing", "easting")
    coords = {"easting": easting, "northing": northing}
    grid = xr.DataArray(
        grid,
        coords=coords,
        dims=dims,
        attrs=header,
    )
    return grid


def _read_header(header_bytes):
    """
    Read GRD file header

    Parameters
    ----------
    header_bytes : byte
        A sequence of 512 bytes containing the header of a
        GRD file.

    Returns
    -------
    header : dict
        Dictionary containing the information present in the
        header.

    Notes
    -----
    The GRD header consists in 512 contiguous bytes.
    It's divided in four sections:

    * Data Storage
    * Geographic Information
    * Data (Z) Scaling
    * Undefined Application Parameters

    """
    header = {}
    # Read data storage
    ES, SF, NE, NV, KX = array.array("i", header_bytes[0 : 16 + 4])  # noqa: E203, N806
    header.update(
        {
            "n_bytes_per_element": ES,
            "sign_flag": SF,
            "shape_e": NE,
            "shape_v": NV,
            "ordering": KX,
        }
    )
    # Read geographic info
    DE, DV, X0, Y0, ROT = array.array(  # noqa: N806
        "d", header_bytes[20 : 52 + 8]  # noqa: E203
    )
    header.update(
        {
            "spacing_e": DE,
            "spacing_v": DV,
            "x_origin": X0,
            "y_origin": Y0,
            "rotation": ROT,
        }
    )
    # Read data scaling
    ZBASE, ZMULT = array.array("d", header_bytes[60 : 68 + 8])  # noqa: E203, N806
    header.update(
        {
            "base_value": ZBASE,
            "data_factor": ZMULT,
        }
    )
    # Read optional parameters
    LABEL = array.array("u", header_bytes[76 : 76 + 48])  # noqa: E203, N806
    MAPNO = array.array("u", header_bytes[124 : 124 + 16])  # noqa: E203, N806
    PROJ, UNITX, UNITY, UNITZ, NVPTS = array.array(  # noqa: N806
        "i", header_bytes[140 : 156 + 4]  # noqa: E203
    )
    IZMIN, IZMAX, IZMED, IZMEA = array.array(  # noqa: N806
        "f", header_bytes[160 : 172 + 4]  # noqa: E203
    )
    (ZVAR,) = array.array("d", header_bytes[176 : 176 + 8])  # noqa: E203, N806
    (PRCS,) = array.array("i", header_bytes[184 : 184 + 4])  # noqa: E203, N806
    header.update(
        {
            "grid_label": LABEL,
            "map_number": MAPNO,
            "map_projection": PROJ,
            "units_x": UNITX,
            "units_y": UNITY,
            "units_z": UNITZ,
            "n_valid_points": NVPTS,
            "grid_min": IZMIN,
            "grid_max": IZMAX,
            "grid_median": IZMED,
            "grid_mean": IZMEA,
            "grid_variance": ZVAR,
            "process_flag": PRCS,
        }
    )
    return header


def _check_ordering(ordering):
    """
    Check if the ordering value is within the ones we are supporting
    """
    if ordering not in (-1, 1):
        raise NotImplementedError(
            f"Found an ordering (a.k.a as KX) equal to '{ordering}'. "
            + "Only orderings equal to 1 and -1 are supported."
        )


def _check_rotation(rotation):
    """
    Check if the rotation value is the one we are supporting
    """
    if rotation != 0:
        raise NotImplementedError(
            f"The grid is rotated '{rotation}' degrees. "
            + "Only unrotated grids are supported."
        )


def _check_sign_flag(sign_flag):
    """
    Check if sign_flag value is within the ones we are supporting
    """
    if sign_flag == 3:
        raise NotImplementedError(
            "Reading .grd files with colour grids is not currenty supported."
        )


def _check_n_bytes_per_element(n_bytes_per_element):
    """
    Check if n_bytes_per_element value is within the ones we are supporting
    """
    if n_bytes_per_element not in (1, 2, 4, 8):
        raise NotImplementedError(
            "Found a 'Grid data element size' (a.k.a. 'ES') value "
            + f"of '{n_bytes_per_element}'. "
            "Compressed .grd files are not currently supported."
        )


def _get_data_type(n_bytes_per_element, sign_flag):
    """
    Return the data type for the grid values

    References
    ----------
    https://docs.python.org/3/library/array.html
    """
    # Determine the data type of the grid elements
    if n_bytes_per_element == 1:
        if sign_flag == 0:
            data_type = "B"  # unsigned char
        elif sign_flag == 1:
            data_type = "b"  # signed char
    elif n_bytes_per_element == 2:
        if sign_flag == 0:
            data_type = "H"  # unsigned short
        elif sign_flag == 1:
            data_type = "h"  # signed short
    elif n_bytes_per_element == 4:
        if sign_flag == 0:
            data_type = "I"  # unsigned int
        elif sign_flag == 1:
            data_type = "i"  # signed int
        elif sign_flag == 2:
            data_type = "f"  # float
    elif n_bytes_per_element == 8:
        data_type = "d"
    return data_type


def _remove_dummies(grid, data_type):
    """
    Replace dummy values for NaNs
    """
    # Create dictionary with dummy value for each data type
    dummies = {
        "b": -127,
        "B": 255,
        "h": -32767,
        "H": 65535,
        "i": -2147483647,
        "I": 4294967295,
        "f": -1e32,
        "d": -1e32,
    }
    if data_type in ("f", "d"):
        grid[grid <= dummies[data_type]] = np.nan
        return grid
    grid[grid == dummies[data_type]] = np.nan
    return grid


def _build_coordinates(west, south, shape, spacing):
    """
    Create the coordinates for the grid

    Generates 1d arrays for the easting and northing coordinates of the grid.
    Assumes unrotated grids.

    Parameters
    ----------
    west : float
        Westernmost coordinate of the grid.
    south : float
        Southernmost coordinate of the grid.
    shape : tuple
        Tuple of ints containing the number of elements along each direction in
        the following order: ``n_northing``, ``n_easting``
    spacing : tuple
        Tuple of floats containing the distance between adjacent grid elements
        along each direction in the following order: ``s_northing``,
        ``s_easting``.

    Returns
    -------
    easting : 1d-array
        Array containing the values of the easting coordinates of the grid.
    northing : 1d-array
        Array containing the values of the northing coordinates of the grid.
    """
    easting = np.linspace(west, west + spacing[1] * (shape[1] - 1), shape[1])
    northing = np.linspace(south, south + spacing[0] * (shape[0] - 1), shape[0])
    return easting, northing
