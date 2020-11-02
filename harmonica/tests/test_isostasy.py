"""
Testing isostasy calculation
"""
import xarray as xr
import numpy as np
import numpy.testing as npt

from ..isostasy import isostasy_airy


def test_isostasy_airy_zero_topography():
    "Root should be zero for zero topography"
    topography = np.zeros(20, dtype=np.float64)
    npt.assert_equal(isostasy_airy(topography, reference_depth=0), 0)
    npt.assert_equal(isostasy_airy(topography, reference_depth=30e3), 30e3)
    # Check that the shape of the topography is preserved
    topography = np.zeros((20, 31), dtype=np.float64)
    assert isostasy_airy(topography).shape == topography.shape
    npt.assert_equal(isostasy_airy(topography, reference_depth=0), 0)
    npt.assert_equal(isostasy_airy(topography, reference_depth=30e3), 30e3)


def test_isostasy_airy():
    "Use a simple integer topography to check the calculations"
    topography = np.array([-2, -1, 0, 1, 2, 3])
    true_root = np.array([-0.5, -0.25, 0, 0.5, 1, 1.5])
    root = isostasy_airy(
        topography,
        density_crust=1,
        density_mantle=3,
        density_water=0.5,
        reference_depth=0,
    )
    npt.assert_equal(root, true_root)


def test_isostasy_airy_dataarray():
    "Pass in a DataArray and make sure things work"
    topography = xr.DataArray(
        np.array([-2, -1, 0, 1, 2, 3]), coords=(np.arange(6),), dims=["something"]
    )
    true_root = np.array([-0.5, -0.25, 0, 0.5, 1, 1.5])
    root = isostasy_airy(
        topography,
        density_crust=1,
        density_mantle=3,
        density_water=0.5,
        reference_depth=0,
    )
    assert isinstance(root, xr.DataArray)
    npt.assert_equal(root.values, true_root)
