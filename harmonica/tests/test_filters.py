# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions from the filter module
"""
import pytest
import numpy as np
import xarray as xr
import xarray.testing as xrt
from verde import grid_coordinates, make_xarray_grid

from ..filters.fft import fft, ifft


@pytest.fixture(name="sample_grid")
def fixture_sample_grid():
    """
    Return a sample grid as an :class:`xarray.DataArray`
    """
    region = (-4e3, 9e3, 10e3, 25e3)
    easting, northing = grid_coordinates(region, spacing=500)
    data = np.sin(easting / 1e3) + np.cos(northing / 1e3)
    return make_xarray_grid((easting, northing), data, data_names="sample").sample


def test_round_trip(sample_grid):
    """
    Test if the wrapped fft and ifft functions satisfy a round trip
    """
    xrt.assert_allclose(sample_grid, ifft(fft(sample_grid)))
