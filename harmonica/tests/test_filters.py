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


@pytest.fixture(name="region")
def fixture_region():
    """
    Return a sample region
    """
    return (-4e3, 9e3, 10e3, 25e3)


@pytest.fixture(name="sample_grid")
def fixture_sample_grid(region):
    """
    Return a sample grid as an :class:`xarray.DataArray`
    """
    easting, northing = grid_coordinates(region, spacing=500)
    data = np.sin(easting / 1e3) + np.cos(northing / 1e3)
    return make_xarray_grid((easting, northing), data, data_names="sample").sample


@pytest.fixture(name="sample_grid_upward")
def fixture_sample_grid_upward(region):
    """
    Return a sample grid as an :class:`xarray.DataArray` with upward coord
    """
    easting, northing, upward = grid_coordinates(region, spacing=500, extra_coords=100)
    data = np.sin(easting / 1e3) + np.cos(northing / 1e3)
    return make_xarray_grid(
        (easting, northing, upward),
        data,
        data_names="sample",
        extra_coords_names="upward",
    ).sample


@pytest.fixture(name="sample_grid_multiple_coords")
def fixture_sample_grid_multiple_coords(region):
    """
    Return a sample grid as an :class:`xarray.DataArray` with multiple coords
    """
    easting, northing, upward = grid_coordinates(region, spacing=500, extra_coords=100)
    data = np.sin(easting / 1e3) + np.cos(northing / 1e3)
    easting, northing = np.unique(easting), np.unique(northing)
    easting_km, northing_km = easting * 1e-3, northing * 1e-3
    dims = ("northing", "easting")
    return xr.DataArray(
        data,
        coords={
            "easting": easting,
            "northing": northing,
            "easting_km": ("easting", easting_km),
            "northing_km": ("northing", northing_km),
            "upward": (dims, upward),
        },
        dims=dims,
    )


def test_fft_round_trip(sample_grid):
    """
    Test if the wrapped fft and ifft functions satisfy a round trip
    """
    xrt.assert_allclose(sample_grid, ifft(fft(sample_grid)))


def test_fft_round_trip_upward(sample_grid_upward):
    """
    Test if the wrapped fft and ifft functions satisfy a round trip
    """
    round_trip = ifft(fft(sample_grid_upward))
    assert "upward" not in round_trip
    # Assert if both arrays are close enough, but dropping upward first
    xrt.assert_allclose(
        sample_grid_upward.drop("upward"), ifft(fft(sample_grid_upward))
    )


def test_fft_no_drop_bad_coords(sample_grid_upward):
    """
    Check if no dropping bad coordinates raises ValueError on upward coord

    ``xrft`` complains when *bad coordinates* are present in the input array.
    This test, along with the ``drop_bad_coordinates`` argument, should be
    removed if ``xrft`` changes this behaviour
    """
    with pytest.raises(ValueError):
        fft(sample_grid_upward, drop_bad_coords=False)


def test_fft_no_drop_bad_coords_multi(sample_grid_multiple_coords):
    """
    Check if no dropping bad coordinates raises ValueError on multiple coords

    This test should fail because ``xrft`` complains when *bad coordinates* are
    present in the input array.
    This test, along with the ``drop_bad_coordinates`` argument, should be
    removed if ``xrft`` changes this behaviour
    """
    with pytest.raises(ValueError):
        fft(sample_grid_multiple_coords, drop_bad_coords=False)
