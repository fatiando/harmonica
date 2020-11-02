"""
Test the gravity correction functions (normal gravity, Bouguer, etc).
"""
import xarray as xr
import numpy as np
import numpy.testing as npt

from ..gravity_corrections import bouguer_correction
from ..constants import GRAVITATIONAL_CONST


def test_bouguer_correction():
    "Test the Bouguer correction using easy to calculate values"
    topography = np.linspace(-10, 20, 100)
    # With these densities, the correction should be equal to the topography
    rhoc = 1 / (1e5 * 2 * np.pi * GRAVITATIONAL_CONST)
    rhow = 0
    bouguer = bouguer_correction(topography, density_crust=rhoc, density_water=rhow)
    assert bouguer.shape == topography.shape
    npt.assert_allclose(bouguer, topography)
    # Check that the shape is preserved for 2D arrays
    bouguer = bouguer_correction(
        topography.reshape(10, 10), density_crust=rhoc, density_water=rhow
    )
    assert bouguer.shape == (10, 10)
    npt.assert_allclose(bouguer, topography.reshape(10, 10))


def test_bouguer_correction_zero_topo():
    "Bouguer correction for zero topography should be zero"
    npt.assert_allclose(bouguer_correction(np.zeros(20)), 0)


def test_bouguer_correction_xarray():
    "Should work the same for an xarray input"
    topography = xr.DataArray(
        np.linspace(-10, 20, 100).reshape((10, 10)),
        coords=(np.arange(10), np.arange(10)),
        dims=("x", "y"),
    )
    # With these densities, the correction should be equal to the topography
    rhoc = 1 / (1e5 * 2 * np.pi * GRAVITATIONAL_CONST)
    rhow = 0
    bouguer = bouguer_correction(topography, density_crust=rhoc, density_water=rhow)
    assert isinstance(bouguer, xr.DataArray)
    assert bouguer.shape == topography.shape
    npt.assert_allclose(bouguer.values, topography.values)
