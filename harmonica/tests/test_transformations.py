# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions for regular grid transformations
"""
import numpy as np
import pytest
import verde as vd
import xrft

from .. import point_gravity
from ..transformations import (
    derivative_easting,
    derivative_northing,
    derivative_upward,
    gaussian_highpass,
    gaussian_lowpass,
    pseudo_gravity,
    reduction_to_pole,
    upward_continuation,
)

from .utils import root_mean_square_error


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


def test_derivative_easting(sample_potential, sample_g_e):
    """
    Test derivative_easting function against the synthetic model
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
    assert rms / np.abs(g_e).max() < 0.015


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
    assert rms / np.abs(g_ee).max() < 0.015


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
    assert rms / np.abs(g_n).max() < 0.015


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
    assert rms / np.abs(g_nn).max() < 0.015


def test_laplace(sample_potential):
    """
    Test second order of derivative fullfill laplace equation
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
    second_deriv_zz = derivative_upward(potential_padded, order=2)
    second_deriv_ee = derivative_easting(potential_padded, order=2)
    second_deriv_nn = derivative_northing(potential_padded, order=2)
    second_deriv_zz = xrft.unpad(second_deriv_zz, pad_width)
    second_deriv_ee = xrft.unpad(second_deriv_ee, pad_width)
    second_deriv_nn = xrft.unpad(second_deriv_nn, pad_width)
    # Compare g_nn + g_ee against -g_zz (trim the borders to ignore boundary
    # effects)
    trim = 6
    second_deriv_sum = second_deriv_ee + second_deriv_nn
    second_deriv_sum = second_deriv_sum[trim:-trim, trim:-trim]
    second_deriv_zz = second_deriv_zz[trim:-trim, trim:-trim]
    rms = root_mean_square_error(second_deriv_sum, -second_deriv_zz)
    assert rms < 1e-20


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
    continuation = upward_continuation(gravity_padded, -10e3)
    continuation = xrft.unpad(continuation, pad_width)
    # Compare against g_z_upward (trim the borders to ignore boundary effects)
    trim = 6
    continuation = continuation[trim:-trim, trim:-trim]
    g_z_upward = sample_g_z_upward[trim:-trim, trim:-trim]
    rms = root_mean_square_error(continuation, g_z_upward)
    assert rms / np.abs(g_z_upward).max() < 0.015
