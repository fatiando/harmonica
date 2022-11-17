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

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

from .. import load_oasis_montaj_grid
from .._io.oasis_montaj_grd import _check_ordering, _check_sign_flag

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


class TestOasisMontajGrid:
    """
    Test if load_oasis_montaj_grid reads grid properly
    """

    expected_grid = xr.load_dataarray(TEST_DATA_DIR / "om_expected.nc")
    atol = 1e-8

    @pytest.mark.parametrize(
        "grd_fname",
        (
            "om_float.grd",
            "om_short.grd",
            "om_long.grd",
            "om_double.grd",
            "om_order.grd",
            "om_compress.grd",
        ),
    )
    def test_simple_grid(self, grd_fname):
        """
        Test a grid with floats
        """
        fname = TEST_DATA_DIR / grd_fname
        grid = load_oasis_montaj_grid(fname)
        atol = self.atol
        if "short" in grd_fname:
            atol = 1e-3
        xrt.assert_allclose(grid, self.expected_grid, atol=atol)

    def test_rotated_grid(self):
        """
        Test loading a rotated grid
        """
        fname = TEST_DATA_DIR / "om_rotate.grd"
        grid = load_oasis_montaj_grid(fname)
        # Check if the rotation angle was correctly read
        assert grid.attrs["rotation"] == -30
        # Check if the values in the grid are the same as the expected grid
        npt.assert_allclose(grid.values, self.expected_grid, atol=self.atol)
        # Check origin of the rotated coordinates
        npt.assert_allclose(grid.easting[0, 0], grid.attrs["x_origin"])
        npt.assert_allclose(grid.northing[0, 0], grid.attrs["y_origin"])
        # Check if we can recover the rotation angle from the coordinates
        west, east = grid.easting[:, 0], grid.easting[:, -1]
        south, north = grid.northing[:, 0], grid.northing[:, -1]
        npt.assert_allclose(
            grid.rotation, np.degrees(np.arctan((north - south) / (east - west)))
        )
