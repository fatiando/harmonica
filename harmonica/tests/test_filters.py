# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions from the filter module.
"""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt
from verde import grid_coordinates, make_xarray_grid

from ..filters._fft import fft, ifft
from ..filters._filters import (
    derivative_easting_kernel,
    derivative_northing_kernel,
    derivative_upward_kernel,
    gaussian_highpass_kernel,
    gaussian_lowpass_kernel,
    reduction_to_pole_kernel,
    upward_continuation_kernel,
)
from ..filters._utils import apply_filter


@pytest.fixture(name="region")
def fixture_region():
    """
    Return a sample region.
    """
    return (-4e3, 9e3, 10e3, 25e3)


@pytest.fixture(name="sample_grid")
def fixture_sample_grid(region):
    """
    Return a sample grid as an :class:`xarray.DataArray`.
    """
    easting, northing = grid_coordinates(region, spacing=500)
    data = np.sin(easting / 1e3) + np.cos(northing / 1e3)
    return make_xarray_grid((easting, northing), data, data_names="sample").sample


@pytest.fixture(name="sample_grid_upward")
def fixture_sample_grid_upward(region):
    """
    Return a sample grid as an :class:`xarray.DataArray` with upward coord.
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
    Return a sample grid as an :class:`xarray.DataArray` with multiple coords.
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
    Test if the wrapped fft and ifft functions satisfy a round trip.
    """
    xrt.assert_allclose(sample_grid, ifft(fft(sample_grid)))


def test_fft_round_trip_upward(sample_grid_upward):
    """
    Test if the wrapped fft and ifft functions satisfy a round trip.
    """
    round_trip = ifft(fft(sample_grid_upward))
    assert "upward" not in round_trip
    # Assert if both arrays are close enough, but dropping upward first
    xrt.assert_allclose(
        sample_grid_upward.drop("upward"), ifft(fft(sample_grid_upward))
    )


def test_fft_no_drop_bad_coords(sample_grid_upward):
    """
    Check if no dropping bad coordinates raises ValueError on upward coord.

    ``xrft`` complains when *bad coordinates* are present in the input array.
    This test, along with the ``drop_bad_coordinates`` argument, should be
    removed if ``xrft`` changes this behaviour
    """
    with pytest.raises(ValueError):
        fft(sample_grid_upward, drop_bad_coords=False)


def test_fft_no_drop_bad_coords_multi(sample_grid_multiple_coords):
    """
    Check if no dropping bad coordinates raises ValueError on multiple coords.

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
    Implement a dummy filter in frequency domain for testing purposes.

    Return an array full of zeroes
    """
    return fourier_transform * 0


def test_apply_filter(sample_grid):
    """
    Test apply_filter function using the dummy_filter.
    """
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
    Check if apply_filter raises error on grid with single dimension.
    """
    with pytest.raises(ValueError, match="Invalid grid with 1 dimensions."):
        apply_filter(invalid_grid_single_dim, dummy_filter)


def test_apply_filter_grid_three_dimensions(invalid_grid_3_dims):
    """
    Check if apply_filter raises error on grid with three dimensions.
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
    Check if apply_filter raises error on grid with single dimension.
    """
    with pytest.raises(ValueError, match="Found nan"):
        apply_filter(invalid_grid_with_nans, dummy_filter)


def test_coordinate_rounding_fix(sample_grid):
    """
    Check that the transformed grid has the same coordinates as the input grid.
    """
    # Apply the dummy filter
    filtered_grid = apply_filter(sample_grid, dummy_filter)

    # Compare coordinates of original grid with coordinates of filtered grid
    npt.assert_array_equal(filtered_grid.easting.values, sample_grid.easting.values)
    npt.assert_array_equal(filtered_grid.northing.values, sample_grid.northing.values)


# -----------------------------
# Test upward derivative filter
# -----------------------------


@pytest.fixture(name="sample_fft_grid")
def fixture_sample_fft_grid():
    """
    Returns a sample fft_grid to be used in test functions.
    """
    domain = (-9e-4, 9e-4, -8e-4, 8e-4)
    freq_easting, freq_northing = grid_coordinates(region=domain, spacing=8e-4)
    dummy_fft = np.ones_like(freq_easting)
    fft_grid = make_xarray_grid(
        (freq_easting, freq_northing),
        dummy_fft,
        data_names=["sample_fft"],
        dims=("freq_northing", "freq_easting"),
    )
    return fft_grid.sample_fft


@pytest.mark.parametrize("order", [1, 2, 3])
def test_derivative_upward_kernel(sample_fft_grid, order):
    """
    Check if derivative_upward_kernel works as expected.
    """
    # Load pre-computed outcome
    expected = (
        -np.array(
            [
                [0.00756596, 0.00565487, 0.00756596],
                [0.00502655, 0.0, 0.00502655],
                [0.00756596, 0.00565487, 0.00756596],
            ]
        )
    ) ** order
    # Check if the filter returns the expected output
    npt.assert_allclose(
        expected, derivative_upward_kernel(sample_fft_grid, order=order), rtol=2e-6
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_derivative_easting_kernel(sample_fft_grid, order):
    """
    Check if derivative_easting_kernel works as expected.
    """
    # Load pre-computed outcome
    expected = np.array([-0.0 - 0.00565487j, 0.0 + 0.0j, 0.0 + 0.00565487j]) ** order
    # Check if the filter returns the expected output
    npt.assert_allclose(
        expected, derivative_easting_kernel(sample_fft_grid, order=order), rtol=2e-6
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_derivative_northing_kernel(sample_fft_grid, order):
    """
    Check if derivative_northing_kernel works as expected.
    """
    # Load pre-computed outcome
    expected = np.array([-0.0 - 0.00502655j, 0.0 + 0.0j, 0.0 + 0.00502655j]) ** order
    # Check if the filter returns the expected output
    npt.assert_allclose(
        expected, derivative_northing_kernel(sample_fft_grid, order=order), rtol=2e-6
    )


@pytest.mark.parametrize("height_displacement", [10, 100, 1000])
def test_upward_continuation_kernel(sample_fft_grid, height_displacement):
    """
    Check if upward_continuation_kernel works as expected.
    """
    # Load pre-computed outcome
    k = np.array(
        [
            [0.00756596, 0.00565487, 0.00756596],
            [0.00502655, 0.0, 0.00502655],
            [0.00756596, 0.00565487, 0.00756596],
        ]
    )
    expected = np.exp(-k * height_displacement)
    # Check if the filter returns the expected output
    npt.assert_allclose(
        expected,
        upward_continuation_kernel(
            sample_fft_grid, height_displacement=height_displacement
        ),
        rtol=3.5e-6,
    )


def test_gaussian_lowpass_kernel(sample_fft_grid, wavelength=10):
    """
    Check if gaussian_lowpass_kernel works as expected.
    """
    # Load pre-computed outcome
    expected = np.array(
        [
            [0.9999275, 0.9999595, 0.9999275],
            [0.999968, 1.0, 0.999968],
            [0.9999275, 0.9999595, 0.9999275],
        ]
    )
    # Check if the filter returns the expected output
    npt.assert_allclose(
        expected,
        gaussian_lowpass_kernel(sample_fft_grid, wavelength=wavelength),
        rtol=2e-6,
    )


def test_gaussian_highpass_kernel(sample_fft_grid, wavelength=100):
    """
    Check if gaussian_highpass_kernel works as expected.
    """
    # Load pre-computed outcome
    expected = np.array(
        [
            [0.00722378, 0.00404181, 0.00722378],
            [0.00319489, 0.0, 0.00319489],
            [0.00722378, 0.00404181, 0.00722378],
        ]
    )
    # Check if the filter returns the expected output
    npt.assert_allclose(
        expected,
        gaussian_highpass_kernel(sample_fft_grid, wavelength=wavelength),
        rtol=2e-6,
    )


def test_reduction_to_pole_kernel(
    sample_fft_grid,
    inclination=60,
    declination=45,
    magnetization_inclination=45,
    magnetization_declination=50,
):
    """
    Check if reduction_to_pole_kernel works as same as the old Fatiando package.
    """
    # Transform degree to rad
    [inclination, declination] = np.deg2rad([inclination, declination])
    [magnetization_inclination, magnetization_declination] = np.deg2rad(
        [magnetization_inclination, magnetization_declination]
    )
    # Calculate expected outcome
    k_easting = 2 * np.pi * sample_fft_grid.freq_easting
    k_northing = 2 * np.pi * sample_fft_grid.freq_northing
    fx, fy, fz = [
        np.cos(inclination) * np.sin(declination),
        np.cos(inclination) * np.cos(declination),
        np.sin(inclination),
    ]
    mx, my, mz = [
        np.cos(magnetization_inclination) * np.sin(magnetization_declination),
        np.cos(magnetization_inclination) * np.cos(magnetization_declination),
        np.sin(magnetization_inclination),
    ]

    a1 = mz * fz - mx * fx
    a2 = mz * fz - my * fy
    a3 = -my * fx - mx * fy
    b1 = mx * fz + mz * fx
    b2 = my * fz + mz * fy

    expected = (k_northing**2 + k_easting**2) / (
        a1 * k_easting**2
        + a2 * k_northing**2
        + a3 * k_easting * k_northing
        + 1j
        * np.sqrt(k_northing**2 + k_easting**2)
        * (b1 * k_easting + b2 * k_northing)
    )
    expected.loc[{"freq_northing": 0, "freq_easting": 0}] = 0
    # Check if the filter returns the expected output
    xrt.assert_allclose(
        expected,
        reduction_to_pole_kernel(
            sample_fft_grid,
            inclination=60,
            declination=45,
            magnetization_inclination=45,
            magnetization_declination=50,
        ),
    )


@pytest.mark.parametrize(
    ("magnetization_inclination", "magnetization_declination"), [(None, 1), (1, None)]
)
def test_invalid_magnetization_angles(
    sample_fft_grid, magnetization_inclination, magnetization_declination
):
    """
    Test if reduction to the pole raise errors when
    invalid magnetization angles are passed.
    """
    if magnetization_inclination is None:
        offender = "magnetization_inclination"
    if magnetization_declination is None:
        offender = "magnetization_declination"
    msg = f"Invalid magnetization degrees. Found `{offender}` as "
    inclination, declination = 1, 30
    with pytest.raises(ValueError, match=msg):
        reduction_to_pole_kernel(
            sample_fft_grid,
            inclination,
            declination,
            magnetization_inclination,
            magnetization_declination,
        )
