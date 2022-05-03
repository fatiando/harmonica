"""
Test functions for spatial derivatives based on FFT
"""
import os
from pathlib import Path

import numpy as np
import xarray as xr
import pytest
import xrft

from .utils import root_mean_square_error
from ..derivatives import derivative_upward

MODULE_DIR = Path(os.path.dirname(__file__))
TEST_DATA_DIR = MODULE_DIR / "data"


@pytest.fixture(name="sample_gravity_grid")
def fixture_sample_gravity_grid():
    """
    Return a sample gravity grid with potential and acceleration components
    """
    # Read grid from file
    fname = TEST_DATA_DIR / "sample-gravity-grid.nc"
    grid = xr.load_dataset(fname)
    return grid


def test_derivative_upward(sample_gravity_grid):
    """
    Test derivative_upward function against the sample grid
    """
    potential = sample_gravity_grid.potential
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": potential.easting.size // 3,
        "northing": potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(potential.drop_vars("upward"), pad_width=pad_width)
    # Calculate upward derivative and unpad it
    derivative = derivative_upward(potential_padded)
    derivative = xrft.unpad(derivative, pad_width)
    # Compare against g_z (trim the borders to ignore boundary effects)
    trim = 5
    derivative = derivative[trim:-trim, trim:-trim]
    g_z = sample_gravity_grid.g_z[trim:-trim, trim:-trim] * 1e-5  # convert to SI units
    rms = root_mean_square_error(derivative, g_z)
    assert rms / np.abs(g_z).max() < 0.015


def test_derivative_upward_order_2(sample_gravity_grid):
    """
    Test higher order of derivative_upward function against the sample grid
    """
    potential = sample_gravity_grid.potential
    # Pad the potential field grid to improve accuracy
    pad_width = {
        "easting": potential.easting.size // 3,
        "northing": potential.northing.size // 3,
    }
    # need to drop upward coordinate (bug in xrft)
    potential_padded = xrft.pad(potential.drop_vars("upward"), pad_width=pad_width)
    # Calculate second upward derivative and unpad it
    second_deriv = derivative_upward(potential_padded, order=2)
    second_deriv = xrft.unpad(second_deriv, pad_width)
    # Compare against g_zz (trim the borders to ignore boundary effects)
    trim = 10
    second_deriv = second_deriv[trim:-trim, trim:-trim]
    g_zz = (
        sample_gravity_grid.g_zz[trim:-trim, trim:-trim] * 1e-9
    )  # convert to SI units
    rms = root_mean_square_error(second_deriv, g_zz)
    assert rms / np.abs(g_zz).max() < 0.015
