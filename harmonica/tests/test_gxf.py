# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# Dataset for testing provided by USGS
# Kucks, R.P., and Hill, P.L., 2000, Wyoming aeromagnetic and gravity maps and data—
# A web site for distribution of data:
# U.S. Geological Survey Open-File Report 00-0198, https://pubs.usgs.gov/of/2000/ofr-00-0198/html/wyoming.htm

"""
Test functions for reading GXF (Grid eXchange Format) files
"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

from .. import read_gxf, read_gxf_raw, gxf_to_xarray

# Define the locations of test data
MODULE_DIR = Path(__file__).parent
TEST_DATA_DIR = MODULE_DIR / "data"
TEST_GXF = TEST_DATA_DIR / "mag5001_gxf"

class TestGXFReader:
    """
    Test if GXF reader functions work properly with USGS modified format
    """
    atol = 1e-8  # Absolute tolerance for floating point comparisons
    
    def test_basic_dimensions(self):
        """
        Test if basic grid dimensions are correct
        """
        grid = gxf_to_xarray(TEST_GXF)
        assert grid.attrs["nx"] == 132
        assert grid.attrs["ny"] == 148
        assert grid.values.shape == (148, 132)

    def test_grid_spacing(self):
        """
        Test grid spacing parameters
        """
        grid = gxf_to_xarray(TEST_GXF)
        npt.assert_almost_equal(grid.attrs["x_inc"], 150.0, decimal=3)
        npt.assert_almost_equal(grid.attrs["y_inc"], 150.0, decimal=3)

    def test_grid_origin(self):
        """
        Test grid origin coordinates
        """
        grid = gxf_to_xarray(TEST_GXF)
        npt.assert_almost_equal(grid.attrs["x_min"], 235000.0, decimal=2)
        npt.assert_almost_equal(grid.attrs["y_min"], 378000.0, decimal=2)

    def test_grid_transform(self):
        """
        Test grid transformation parameters
        """
        _, metadata = read_gxf(TEST_GXF)
        transform = metadata["TRANSFORM"].split()
        npt.assert_almost_equal(float(transform[0]), 1.0, decimal=10)
        npt.assert_almost_equal(float(transform[1]), 0.0, decimal=1)

    def test_dummy_value(self):
        """
        Test dummy value specification
        """
        _, metadata = read_gxf(TEST_GXF)
        npt.assert_almost_equal(float(metadata["DUMMY"]), 1.0e30)

    def test_projection_parameters(self):
        """
        Test projection information
        """
        grid = gxf_to_xarray(TEST_GXF)
        
        # Test projection type and units
        assert grid.attrs["projection_type"].strip() == "lambert conformal conic"
        assert grid.attrs["projection_units"].strip() == "meters"
        
        # Test ellipsoid parameters
        npt.assert_almost_equal(grid.attrs["proj_semi_major_axis"], 6378206.40)
        npt.assert_almost_equal(grid.attrs["proj_semi_minor_axis"], 6356583.80)
        
        # Test projection specific parameters
        npt.assert_almost_equal(grid.attrs["proj_reference_longitude"], -107.50000)
        npt.assert_almost_equal(grid.attrs["proj_reference_latitude"], 41.00000)
        npt.assert_almost_equal(grid.attrs["proj_first_standard_parallel"], 33.00000)
        npt.assert_almost_equal(grid.attrs["proj_second_standard_parallel"], 45.00000)
        npt.assert_almost_equal(grid.attrs["proj_false_easting"], 0.00000)
        npt.assert_almost_equal(grid.attrs["proj_false_northing"], 0.00000)

    def test_grid_orientation(self):
        """
        Test grid orientation parameters
        """
        grid = gxf_to_xarray(TEST_GXF)
        assert grid.attrs["ROTATION"] == 0.0
        assert grid.attrs["SENSE"] == 1

    def test_coordinate_generation(self):
        """
        Test if coordinates are correctly generated
        """
        grid = gxf_to_xarray(TEST_GXF)
        
        # Test X coordinates
        expected_x = np.arange(132) * 150.0 + 235000.0
        npt.assert_array_almost_equal(grid.x.values, expected_x)
        
        # Test Y coordinates
        expected_y = np.arange(148) * 150.0 + 378000.0
        npt.assert_array_almost_equal(grid.y.values, expected_y)

    def test_raw_headers(self):
        """
        Test raw header reading
        """
        data_list, headers = read_gxf_raw(TEST_GXF)
        
        assert headers["POINTS"] == "   132"
        assert headers["ROWS"] == "   148"
        assert headers["PTSEPARATION"] == "    150.000"
        assert headers["RWSEPARATION"] == "    150.000"
        assert headers["XORIGIN"] == "     235000.00000000"
        assert headers["YORIGIN"] == "     378000.00000000"
        assert headers["PRJTYPE"].strip() == "lambert conformal conic"

    def test_comment_preservation(self):
        """
        Test if USGS modified format comment is preserved
        """
        grid = gxf_to_xarray(TEST_GXF)
        assert "COMMENT" in grid.attrs
        assert "USGS modified GXF format" in grid.attrs["COMMENT"]

    def test_coordinate_dimensions(self):
        """
        Test if coordinate dimensions match the grid size
        """
        grid = gxf_to_xarray(TEST_GXF)
        assert len(grid.x) == 132
        assert len(grid.y) == 148

    def test_data_type(self):
        """
        Test if the data is loaded as float values
        """
        grid = gxf_to_xarray(TEST_GXF)
        assert np.issubdtype(grid.values.dtype, np.floating)
