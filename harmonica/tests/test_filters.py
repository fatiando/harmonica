# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions from the filter module
"""
import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from verde import grid_coordinates, make_xarray_grid

from ..filters._fft import fft, ifft
from ..filters._filters import derivative_upward_kernel
from ..filters._utils import apply_filter


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


# -------------------------------
# Tests for apply_filter function
# -------------------------------


def dummy_filter(fourier_transform):
    """
    Implement a dummy filter in frequency domain for testing purposes

    Return an array full of zeroes
    """
    return fourier_transform * 0


def test_apply_filter(sample_grid):
    """
    Test apply_filter function using the dummy_filter
    """
    print(sample_grid)
    # Apply the dummy filter
    filtered_grid = apply_filter(sample_grid, dummy_filter)
    # Compare output with expected results
    expected = sample_grid * 0
    xrt.assert_allclose(filtered_grid, expected)


@pytest.fixture(name="invalid_grid_single_dim")
def fixture_invalid_grid_single_dim():
    """
    Return a sample grid with a single dimension.

    This fixture is meant to test if apply_filter raises an error on a grid
    with a single dimension.
    """
    x = np.linspace(0, 10, 11)
    y = x**2
    grid = xr.DataArray(y, coords={"x": x}, dims=("x",))
    return grid


@pytest.fixture(name="invalid_grid_3_dims")
def fixture_invalid_grid_3_dims():
    """
    Return a sample grid with 3 dimensions.

    This fixture is meant to test if apply_filter raises an error on a grid
    with 3 dimensions.
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 9)
    z = np.linspace(20, 30, 5)
    xx, yy, zz = np.meshgrid(x, y, z)
    data = xx + yy + zz
    grid = xr.DataArray(data, coords={"x": x, "y": y, "z": z}, dims=("y", "x", "z"))
    return grid


def test_apply_filter_grid_single_dimension(invalid_grid_single_dim):
    """
    Check if apply_filter raises error on grid with single dimension
    """
    with pytest.raises(ValueError, match="Invalid grid with 1 dimensions."):
        apply_filter(invalid_grid_single_dim, dummy_filter)


def test_apply_filter_grid_three_dimensions(invalid_grid_3_dims):
    """
    Check if apply_filter raises error on grid with single dimension
    """
    with pytest.raises(ValueError, match="Invalid grid with 3 dimensions."):
        apply_filter(invalid_grid_3_dims, dummy_filter)


@pytest.fixture(name="invalid_grid_with_nans")
def fixture_invalid_grid_with_nans(sample_grid):
    """
    Return a sample grid with nans.

    This fixture is meant to test if apply_filter raises an error on a grid
    with a nans.
    """
    sample_grid[2, 4] = np.nan
    return sample_grid


def test_apply_filter_grid_with_nans(invalid_grid_with_nans):
    """
    Check if apply_filter raises error on grid with single dimension
    """
    with pytest.raises(ValueError, match="Found nan"):
        apply_filter(invalid_grid_with_nans, dummy_filter)


# -----------------------------
# Test upward derivative filter
# -----------------------------


@pytest.fixture(name="sample_fft_grid")
def fixture_sample_fft_grid():
    """
    Returns a sample fft_grid to be used in test functions
    """
    domain = (-9e-4, 9e-4, -8e-4, -8e-4)
    freq_easting, freq_northing = grid_coordinates(region=domain, spacing=1e-5)
    dummy_fft = np.ones_like(freq_easting)
    fft_grid = make_xarray_grid(
        (freq_easting, freq_northing),
        dummy_fft,
        data_names=["sample_fft"],
        dims=("freq_northing", "freq_easting"),
    )
    return fft_grid.sample_fft


@pytest.mark.parametrize("order", (1, 2, 3))
def test_derivative_upward_kernel(sample_fft_grid, order):
    """
    Check if derivative_upward_kernel works as expected
    """
    # Calculate expected outcome
    k_easting = 2 * np.pi * sample_fft_grid.freq_easting
    k_northing = 2 * np.pi * sample_fft_grid.freq_northing
    expected = np.sqrt(k_easting**2 + k_northing**2) ** order
    # Check if the filter returns the expected output
    xrt.assert_allclose(
        expected, derivative_upward_kernel(sample_fft_grid, order=order)
    )
