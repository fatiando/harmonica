"""
Test the sample data loading functions.
"""
import numpy.testing as npt

from ..datasets.sample_data import fetch_gravity_earth, fetch_topography_earth


def test_gravity_earth():
    "Sanity checks for the loaded grid"
    grid = fetch_gravity_earth()
    assert grid.gravity.shape == (361, 721)
    npt.assert_allclose(grid.gravity.max(), 9.8018358e05)
    npt.assert_allclose(grid.gravity.min(), 9.7476403e05)
    assert grid.height_over_ell.shape == (361, 721)
    npt.assert_allclose(grid.height_over_ell, 10000)


def test_topography_earth():
    "Sanity checks for the loaded grid"
    grid = fetch_topography_earth()
    assert grid.topography.shape == (361, 721)
    npt.assert_allclose(grid.topography.max(), 5622)
    npt.assert_allclose(grid.topography.min(), -8397)
