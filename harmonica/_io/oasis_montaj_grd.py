# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Function to read Oasis Montaj© .grd file
"""

import array
import zlib

import numpy as np
import xarray as xr

# Define the valid element sizes (ES variable) for GRD files
# (values > 1024 correspond to compressed versions of the grid)
VALID_ELEMENT_SIZES = (1, 2, 4, 8, 1024 + 1, 1024 + 2, 1024 + 4, 1024 + 8)
# Define dummy values for each data type
DUMMIES = {
    "b": -127,
    "B": 255,
    "h": -32767,
    "H": 65535,
    "i": -2147483647,
    "I": 4294967295,
    "f": -1e32,
    "d": -1e32,
}


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

        This function is not supporting orderings different than ±1,
        or colour grids.

    Parameters
    ----------
    fname : string or file-like object
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
        # Get data type for the grid elements
        data_type = _get_data_type(header["n_bytes_per_element"], header["sign_flag"])
        # Read grid
        grid = grd_file.read()
    # Decompress grid if needed
    if header["n_bytes_per_element"] > 1024:
        grid = _decompress_grid(grid)
    # Load the grid values as an array with the proper data_type
    grid = array.array(data_type, grid)
    # Convert to numpy array as float64
    grid = np.array(grid, dtype=np.float64)
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
    # (ignore map LABEL and MAPNO)
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


def _get_data_type(n_bytes_per_element, sign_flag):
    """
    Return the data type for the grid values

    References
    ----------
    https://docs.python.org/3/library/array.html
    """
    # Check if number of bytes per element is valid
    if n_bytes_per_element not in VALID_ELEMENT_SIZES:
        raise NotImplementedError(
            "Found a 'Grid data element size' (a.k.a. 'ES') value "
            + f"of '{n_bytes_per_element}'. "
            "Only values equal to 1, 2, 4 and 8 are valid, "
            + "along with their compressed counterparts (1025, 1026, 1028, 1032)."
        )
    # Shift the n_bytes_per_element in case of compressed grids
    if n_bytes_per_element > 1024:
        n_bytes_per_element -= 1024
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
    if data_type in ("f", "d"):
        grid[grid <= DUMMIES[data_type]] = np.nan
        return grid
    grid[grid == DUMMIES[data_type]] = np.nan
    return grid


def _decompress_grid(grid_compressed):
    """
    Decompress the grid using gzip

    Even if the header specifies that the grid is compressed using a LZRW1
    algorithm, it's using gzip instead. The first two 4 bytes sequences
    correspond to the compression signature and
    to the compression type. We are going to ignore those and start reading
    from the number of blocks (offset 8).

    Parameters
    ----------
    grid_compressed : bytes
        Sequence of bytes corresponding to the compressed grid. They should be
        every byte starting from offset 512 of the GRD file until its end.

    Returns
    -------
    grid : bytes
        Uncompressed version of the ``grid_compressed`` parameter.
    """
    # Number of blocks
    (n_blocks,) = array.array("i", grid_compressed[8 : 8 + 4])
    # Number of vectors per block
    (vectors_per_block,) = array.array("i", grid_compressed[12 : 12 + 4])
    # File offset from start of every block
    block_offsets = array.array("q", grid_compressed[16 : 16 + n_blocks * 8])
    # Compressed size of every block
    compressed_block_sizes = array.array(
        "i",
        grid_compressed[16 + n_blocks * 8 : 16 + n_blocks * 8 + n_blocks * 4],
    )
    # Combine grid
    grid = b""
    # Read each block
    for i in range(n_blocks):
        # Define the start and end offsets for each compressed blocks
        # We need to remove the 512 to account for the missing header.
        # There is an unexplained 16 byte header that we also need to remove.
        start_offset = block_offsets[i] - 512 + 16
        end_offset = compressed_block_sizes[i] + block_offsets[i] - 512
        # Decompress the block
        grid_sub = zlib.decompress(
            grid_compressed[start_offset:end_offset],
            bufsize=zlib.DEF_BUF_SIZE,
        )
        # Add it to the running grid
        grid += grid_sub
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
