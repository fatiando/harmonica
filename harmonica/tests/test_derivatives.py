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
    potential_padded = xrft.pad(potential.drop("upward"), pad_width=pad_width)
    # Calculate upward derivative and unpad it
    derivative = derivative_upward(potential_padded)
    derivative = xrft.unpad(derivative, pad_width)
    # Compare against g_z (trim the borders to ignore boundary effects)
    trim = 5
    rms = root_mean_square_error(
        derivative[trim:-trim, trim:-trim],
        sample_gravity_grid.g_z[trim:-trim, trim:-trim],
    )
    assert rms / np.abs(sample_gravity_grid.g_z).max() < 0.015
