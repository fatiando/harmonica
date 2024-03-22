# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions for regular grid transformations
"""
from pathlib import Path

import numpy as np
import pytest
import verde as vd
import xarray as xr
import xarray.testing as xrt
import xrft

from .. import point_gravity
from .._transformations import (
    _get_dataarray_coordinate,
    derivative_easting,
    derivative_northing,
    derivative_upward,
    gaussian_highpass,
    gaussian_lowpass,
    reduction_to_pole,
    total_gradient_amplitude,
    upward_continuation,
)
from .utils import root_mean_square_error

MODULE_DIR = Path(__file__).parent
TEST_DATA_DIR = MODULE_DIR / "data"


@pytest.fixture(name="sample_sources")
def fixture_sample_sources():
    """
    Define a pair of sample point sources used to build tests
    """
    points = [
        [-50e3, 50e3],
        [-50e3, 50e3],
        [-30e3, -20e3],
    ]

    masses = [8e8, -3e8]
    return points, masses


@pytest.fixture(name="sample_grid_coords")
def fixture_sample_grid_coords():
    """
    Define sample grid coordinates
    """
    grid_coords = vd.grid_coordinates(
        region=(-150e3, 150e3, -150e3, 150e3), shape=(41, 41), extra_coords=0
    )
    return grid_coords


@pytest.fixture(name="upward_grid_coords")
def fixture_upward_grid_coords():
    """
    Define upward grid coordinates
    """
    grid_coords = vd.grid_coordinates(
        region=(-150e3, 150e3, -150e3, 150e3), shape=(41, 41), extra_coords=10e3
    )
    return grid_coords


@pytest.fixture(name="sample_potential")
def fixture_sample_potential(sample_grid_coords, sample_sources):
    """
    Return gravity potential field of sample sources on sample grid coords
    """
    points, masses = sample_sources
    potential = point_gravity(sample_grid_coords, points, masses, field="potential")
    potential = vd.make_xarray_grid(
        sample_grid_coords,
        potential,
        data_names="potential",
        extra_coords_names="upward",
    )
    return potential.potential


@pytest.fixture(name="sample_g_z")
def fixture_sample_g_z(sample_grid_coords, sample_sources):
    """
    Return g_z field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_z = point_gravity(sample_grid_coords, points, masses, field="g_z")
    g_z = vd.make_xarray_grid(
        sample_grid_coords,
        g_z,
        data_names="g_z",
        extra_coords_names="upward",
    )
    return g_z.g_z


@pytest.fixture(name="sample_g_z_upward")
def fixture_sample_g_z_upward(upward_grid_coords, sample_sources):
    """
    Return g_z field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_z = point_gravity(upward_grid_coords, points, masses, field="g_z")
    g_z = vd.make_xarray_grid(
        upward_grid_coords,
        g_z,
        data_names="g_z",
        extra_coords_names="upward",
    )
    return g_z.g_z


@pytest.fixture(name="sample_g_n")
def fixture_sample_g_n(sample_grid_coords, sample_sources):
    """
    Return g_n field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_n = point_gravity(sample_grid_coords, points, masses, field="g_n")
    g_n = vd.make_xarray_grid(
        sample_grid_coords,
        g_n,
        data_names="g_n",
        extra_coords_names="upward",
    )
    return g_n.g_n


@pytest.fixture(name="sample_g_e")
def fixture_sample_g_e(sample_grid_coords, sample_sources):
    """
    Return g_e field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_e = point_gravity(sample_grid_coords, points, masses, field="g_e")
    g_e = vd.make_xarray_grid(
        sample_grid_coords,
        g_e,
        data_names="g_e",
        extra_coords_names="upward",
    )
    return g_e.g_e


@pytest.fixture(name="sample_g_zz")
def fixture_sample_g_zz(sample_grid_coords, sample_sources):
    """
    Return g_zz field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_zz = point_gravity(sample_grid_coords, points, masses, field="g_zz")
    g_zz = vd.make_xarray_grid(
        sample_grid_coords,
        g_zz,
        data_names="g_zz",
        extra_coords_names="upward",
    )
    return g_zz.g_zz


@pytest.fixture(name="sample_g_nn")
def fixture_sample_g_nn(sample_grid_coords, sample_sources):
    """
    Return g_nn field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_nn = point_gravity(sample_grid_coords, points, masses, field="g_nn")
    g_nn = vd.make_xarray_grid(
        sample_grid_coords,
        g_nn,
        data_names="g_nn",
        extra_coords_names="upward",
    )
    return g_nn.g_nn


@pytest.fixture(name="sample_g_ee")
def fixture_sample_g_ee(sample_grid_coords, sample_sources):
    """
    Return g_ee field of sample points on sample grid coords
    """
    points, masses = sample_sources
    g_ee = point_gravity(sample_grid_coords, points, masses, field="g_ee")
    g_ee = vd.make_xarray_grid(
        sample_grid_coords,
        g_ee,
        data_names="g_ee",
        extra_coords_names="upward",
    )
    return g_ee.g_ee


@pytest.mark.parametrize("index, expected_dimension", ([1, "easting"], [0, "northing"]))
def test_get_dataarray_coordinate(index, expected_dimension, sample_potential):
    """
    Test the _get_dataarray_coordinate private function
    """
    dimension = _get_dataarray_coordinate(sample_potential, index)
    assert dimension == expected_dimension


@pytest.mark.parametrize("index, dimension", ([1, "easting"], [0, "northing"]))
def test_get_dataarray_coordinate_invalid_grid(index, dimension, sample_potential):
    """
    Test if _get_dataarray_coordinate raises error when grid have an additional
    coordinate that share one of the horizontal dimensions
    """
    # Add another horizontal coordinate that shares the same dimension
    extra_coord = np.ones_like(sample_potential[dimension])
    grid = sample_potential.assign_coords({"extra_coord": (dimension, extra_coord)})
    # Check if function raises an error
    err_msg = "Grid contains more than one coordinate along the"
    with pytest.raises(ValueError, match=err_msg):
        _get_dataarray_coordinate(grid, index)


@pytest.mark.parametrize(
    "dimension, derivative_func",
    (["easting", derivative_easting], ["northing", derivative_northing]),
)
def test_horizontal_derivative_with_invalid_grid(
    dimension, derivative_func, sample_potential
):
    """
    Test if the horizontal derivative functions raise an error when passing
    a grid that has an additional coordinate that share the horizontal
    dimension and the "finite-diff" method is selected.
    """
    # Add another horizontal coordinate that shares the same dimension
    extra_coord = np.ones_like(sample_potential[dimension])
    grid = sample_potential.assign_coords({"extra_coord": (dimension, extra_coord)})
    # Check if function raises an error
    err_msg = "Grid contains more than one coordinate along the"
    with pytest.raises(ValueError, match=err_msg):
        derivative_func(grid, method="finite-diff")


def test_derivative_upward(sample_potential, sample_g_z):
    """
    Test derivative_upward function against the synthetic model
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate upward derivative and unpad it
    derivative = derivative_upward(potential_padded)
    derivative = xrft.unpad(derivative, pad_width)
    # Compare against g_z (trim the borders to ignore boundary effects)
    trim = 6
    derivative = derivative[trim:-trim, trim:-trim]
    g_z = sample_g_z[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    rms = root_mean_square_error(derivative, g_z)
    assert rms / np.abs(g_z).max() < 0.015


def test_derivative_upward_order2(sample_potential, sample_g_zz):
    """
    Test higher order of derivative_upward function against the sample grid
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate second upward derivative and unpad it
    second_deriv = derivative_upward(potential_padded, order=2)
    second_deriv = xrft.unpad(second_deriv, pad_width)
    # Compare against g_zz (trim the borders to ignore boundary effects)
    trim = 6
    second_deriv = second_deriv[trim:-trim, trim:-trim]
    g_zz = sample_g_zz[trim:-trim, trim:-trim] * 1e-9  # convert to SI units
    rms = root_mean_square_error(second_deriv, g_zz)
    assert rms / np.abs(g_zz).max() < 0.015


@pytest.mark.parametrize(
    "derivative_func",
    [derivative_easting, derivative_northing],
    ids=["derivative_easting", "derivative_northing"],
)
def test_invalid_method_horizontal_derivatives(sample_potential, derivative_func):
    """
    Test if passing and invalid method to horizontal derivatives raise an error
    """
    method = "bla"
    err_msg = f"Invalid method '{method}'."
    with pytest.raises(ValueError, match=err_msg):
        derivative_func(sample_potential, method=method)


def test_derivative_easting_finite_diff(sample_potential, sample_g_e):
    """
    Test derivative_easting function against the synthetic model using finite
    differences
    """
    # Calculate easting derivative
    derivative = derivative_easting(sample_potential)
    # Compare against g_e
    g_e = sample_g_e * 1e-5  # convert to SI units
    rms = root_mean_square_error(derivative, g_e)
    assert rms / np.abs(g_e).max() < 0.01


def test_derivative_easting_finite_diff_order_2(sample_potential, sample_g_ee):
    """
    Test higher order of derivative_easting function against the sample grid
    using finite differences
    """
    # Calculate second easting derivative
    second_deriv = derivative_easting(sample_potential, order=2)
    # Compare against g_e
    g_ee = sample_g_ee * 1e-9  # convert to SI units
    rms = root_mean_square_error(second_deriv, g_ee)
    assert rms / np.abs(g_ee).max() < 0.1


def test_derivative_easting_fft(sample_potential, sample_g_e):
    """
    Test derivative_easting function against the synthetic model using FFTs
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate easting derivative and unpad it
    derivative = derivative_easting(potential_padded)
    derivative = xrft.unpad(derivative, pad_width)
    # Compare against g_e (trim the borders to ignore boundary effects)
    trim = 6
    derivative = derivative[trim:-trim, trim:-trim]
    g_e = sample_g_e[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    rms = root_mean_square_error(derivative, g_e)
    assert rms / np.abs(g_e).max() < 0.1


def test_derivative_easting_order2(sample_potential, sample_g_ee):
    """
    Test higher order of derivative_easting function against the sample grid
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate second easting derivative and unpad it
    second_deriv = derivative_easting(potential_padded, order=2)
    second_deriv = xrft.unpad(second_deriv, pad_width)
    # Compare against g_ee (trim the borders to ignore boundary effects)
    trim = 6
    second_deriv = second_deriv[trim:-trim, trim:-trim]
    g_ee = sample_g_ee[trim:-trim, trim:-trim] * 1e-9  # convert to SI units
    rms = root_mean_square_error(second_deriv, g_ee)
    assert rms / np.abs(g_ee).max() < 0.1


def test_derivative_northing_finite_diff(sample_potential, sample_g_n):
    """
    Test derivative_northing function against the synthetic model using finite
    differences
    """
    # Calculate northing derivative
    derivative = derivative_northing(sample_potential)
    # Compare against g_e
    g_n = sample_g_n * 1e-5  # convert to SI units
    rms = root_mean_square_error(derivative, g_n)
    assert rms / np.abs(g_n).max() < 0.01


def test_derivative_northing_finite_diff_order_2(sample_potential, sample_g_nn):
    """
    Test higher order of derivative_northing function against the sample grid
    using finite differences
    """
    # Calculate second northing derivative
    second_deriv = derivative_northing(sample_potential, order=2)
    # Compare against g_e
    g_nn = sample_g_nn * 1e-9  # convert to SI units
    rms = root_mean_square_error(second_deriv, g_nn)
    assert rms / np.abs(g_nn).max() < 0.1


def test_derivative_northing(sample_potential, sample_g_n):
    """
    Test derivative_northing function against the synthetic model
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate northing derivative and unpad it
    derivative = derivative_northing(potential_padded)
    derivative = xrft.unpad(derivative, pad_width)
    # Compare against g_n (trim the borders to ignore boundary effects)
    trim = 6
    derivative = derivative[trim:-trim, trim:-trim]
    g_n = sample_g_n[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    rms = root_mean_square_error(derivative, g_n)
    assert rms / np.abs(g_n).max() < 0.1


def test_derivative_northing_order2(sample_potential, sample_g_nn):
    """
    Test higher order of derivative_northing function against the sample grid
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate second northing derivative and unpad it
    second_deriv = derivative_northing(potential_padded, order=2)
    second_deriv = xrft.unpad(second_deriv, pad_width)
    # Compare against g_nn (trim the borders to ignore boundary effects)
    trim = 6
    second_deriv = second_deriv[trim:-trim, trim:-trim]
    g_nn = sample_g_nn[trim:-trim, trim:-trim] * 1e-9  # convert to SI units
    rms = root_mean_square_error(second_deriv, g_nn)
    assert rms / np.abs(g_nn).max() < 0.1


def test_laplace_fft(sample_potential):
    """
    Test if second order of derivatives fulfill Laplace equation

    We will use FFT computations only.
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate second northing derivative and unpad it
    method = "fft"
    second_deriv_ee = derivative_easting(potential_padded, order=2, method=method)
    second_deriv_nn = derivative_northing(potential_padded, order=2, method=method)
    second_deriv_zz = derivative_upward(potential_padded, order=2)
    second_deriv_ee = xrft.unpad(second_deriv_ee, pad_width)
    second_deriv_nn = xrft.unpad(second_deriv_nn, pad_width)
    second_deriv_zz = xrft.unpad(second_deriv_zz, pad_width)
    # Compare g_nn + g_ee against -g_zz (trim the borders to ignore boundary
    # effects)
    trim = 6
    second_deriv_sum = second_deriv_ee + second_deriv_nn
    second_deriv_sum = second_deriv_sum[trim:-trim, trim:-trim]
    second_deriv_zz = second_deriv_zz[trim:-trim, trim:-trim]
    xrt.assert_allclose(second_deriv_sum, -second_deriv_zz, atol=1e-20)


def test_upward_continuation(sample_g_z, sample_g_z_upward):
    """
    Test upward_continuation function against the synthetic model
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_g_z.easting.size // 3,
        "northing": sample_g_z.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    gravity_padded = xrft.pad(
        sample_g_z.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate upward continuation and unpad it
    continuation = upward_continuation(gravity_padded, 10e3)
    continuation = xrft.unpad(continuation, pad_width)
    # Compare against g_z_upward (trim the borders to ignore boundary effects)
    trim = 6
    continuation = continuation[trim:-trim, trim:-trim]
    g_z_upward = sample_g_z_upward[trim:-trim, trim:-trim]
    # Drop upward for comparison
    g_z_upward = g_z_upward.drop("upward")
    xrt.assert_allclose(continuation, g_z_upward, atol=1e-8)


def test_total_gradient_amplitude(sample_potential, sample_g_n, sample_g_e, sample_g_z):
    """
    Test total_gradient_amplitude function against the synthetic model
    """
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": sample_potential.easting.size // 3,
        "northing": sample_potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(
        sample_potential.drop_vars("upward"),
        pad_width=pad_width,
    )
    # Calculate total gradient amplitude and unpad it
    tga = total_gradient_amplitude(potential_padded)
    tga = xrft.unpad(tga, pad_width)
    # Compare against g_tga (trim the borders to ignore boundary effects)
    trim = 6
    tga = tga[trim:-trim, trim:-trim]
    g_e = sample_g_e[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    g_n = sample_g_n[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    g_z = sample_g_z[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    g_tga = np.sqrt(g_e**2 + g_n**2 + g_z**2)
    rms = root_mean_square_error(tga, g_tga)
    assert rms / np.abs(g_tga).max() < 0.1


class Testfilter:
    """
    Test filter result against the output from oasis montaj
    """

    expected_grid = xr.open_dataset(TEST_DATA_DIR / "filter.nc")

    def test_gaussian_lowpass_grid(self):
        """
        Test gaussian_lowpass function against the output from oasis montaj
        """
        low_pass = gaussian_lowpass(self.expected_grid.filter_data, 10)
        xrt.assert_allclose(self.expected_grid.filter_lp10, low_pass, atol=1e-6)

    def test_gaussian_highpass_grid(self):
        """
        Test gaussian_highpass function against the output from oasis montaj
        """
        high_pass = gaussian_highpass(self.expected_grid.filter_data, 10)
        xrt.assert_allclose(self.expected_grid.filter_hp10, high_pass, atol=1e-6)

    def test_reduction_to_pole_grid(self):
        """
        Test greduction_to_pole function against the output from oasis montaj
        """
        rtp = reduction_to_pole(self.expected_grid.filter_data, 60, 45)
        # Remove mean value to match OM result
        xrt.assert_allclose(
            self.expected_grid.filter_rtp - self.expected_grid.filter_data.mean(),
            rtp,
            atol=1,
        )
