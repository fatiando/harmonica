# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test function to read Oasis Montaj© .grd file
"""
from pathlib import Path

import pytest
import xarray as xr
import xarray.testing as xrt

from .. import load_oasis_montaj_grid
from .._io.oasis_montaj_grd import (
    _check_ordering,
    _check_rotation,
    _check_sign_flag,
    _check_uncompressed_grid,
)

MODULE_DIR = Path(__file__).parent
TEST_DATA_DIR = MODULE_DIR / "data"


@pytest.mark.parametrize("ordering", (-1, 1))
def test_check_ordering_valid(ordering):
    """
    Test _check_ordering with valid values
    """
    _check_ordering(ordering)


@pytest.mark.parametrize("ordering", (-2, 0, 2))
def test_check_ordering_invalid(ordering):
    """
    Test _check_ordering with invalid values
    """
    with pytest.raises(NotImplementedError, match="Found an ordering"):
        _check_ordering(ordering)


def test_check_rotation_valid():
    """
    Test _check_rotation with valid value
    """
    _check_rotation(0)


@pytest.mark.parametrize("rotation", (-60, 30, 90))
def test_check_rotation_invalid(rotation):
    """
    Test _check_rotation with invalid values
    """
    with pytest.raises(NotImplementedError, match="The grid is rotated"):
        _check_rotation(rotation)


@pytest.mark.parametrize("sign_flag", (0, 1, 2))
def test_check_sign_flag_valid(sign_flag):
    """
    Test _check_sign_flag with valid value
    """
    _check_sign_flag(sign_flag)


def test_check_sign_flag_invalid():
    """
    Test _check_sign_flag with invalid value
    """
    with pytest.raises(NotImplementedError, match="colour grids is not"):
        _check_sign_flag(3)


@pytest.mark.parametrize("n_bytes_per_element", (1, 2, 4, 8))
def test_check_uncompressed_grid_valid(n_bytes_per_element):
    """
    Test _check_uncompressed_grid with valid values
    """
    _check_uncompressed_grid(n_bytes_per_element)


@pytest.mark.parametrize(
    "n_bytes_per_element", (1024 + 1, 1024 + 2, 1024 + 4, 1024 + 8)
)
def test_check_uncompressed_grid_invalid(n_bytes_per_element):
    """
    Test _check_uncompressed_grid with invalid values
    """
    msg = "Compressed .grd files are not currently supported"
    with pytest.raises(NotImplementedError, match=msg):
        _check_uncompressed_grid(n_bytes_per_element)


class TestOasisMontajGrid:
    """
    Test if load_oasis_montaj_grid reads grid properly
    """

    expected_grid = xr.load_dataarray(TEST_DATA_DIR / "om_expected.nc")

    @pytest.mark.parametrize(
        "grd_fname",
        (
            "om_float.grd",
            "om_short.grd",
            "om_long.grd",
            "om_double.grd",
        ),
    )
    def test_simple_grid(self, grd_fname):
        """
        Test a grid with floats
        """
        fname = TEST_DATA_DIR / grd_fname
        grid = load_oasis_montaj_grid(fname)
        atol = 1e-8
        if "short" in grd_fname:
            atol = 1e-3
        xrt.assert_allclose(grid, self.expected_grid, atol=atol)
