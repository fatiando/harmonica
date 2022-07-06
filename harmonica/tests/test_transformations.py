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
from ..transformations import derivative_upward
from .utils import root_mean_square_error


@pytest.fixture(name="sample_sources")
def fixture_sample_sources():
    """
    Define a pair of sample point sources used to build tests
    """
    points = [
        [-50e3, 50e3],
        [-50e3, 50e3],
        [-80e3, -20e3],
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
    Return g_z field of sample prisms on sample grid coords
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


@pytest.fixture(name="sample_g_zz")
def fixture_sample_g_zz(sample_grid_coords, sample_sources):
    """
    Return g_zz field of sample prisms on sample grid coords
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
