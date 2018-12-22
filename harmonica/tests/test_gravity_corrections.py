"""
Test the gravity correction functions (normal gravity, Bouguer, etc).
"""
import xarray as xr
import numpy as np
import numpy.testing as npt

from ..gravity_corrections import normal_gravity, bouguer_correction
from ..ellipsoid import KNOWN_ELLIPSOIDS, set_ellipsoid, get_ellipsoid
from ..constants import GRAVITATIONAL_CONST


def test_normal_gravity():
    "Compare normal gravity values at pole and equator"
    rtol = 1e-10
    height = 0
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            # Convert gamma to mGal
            gamma_pole = get_ellipsoid().gravity_pole * 1e5
            gamma_eq = get_ellipsoid().gravity_equator * 1e5
            npt.assert_allclose(gamma_pole, normal_gravity(-90, height), rtol=rtol)
            npt.assert_allclose(gamma_pole, normal_gravity(90, height), rtol=rtol)
            npt.assert_allclose(gamma_eq, normal_gravity(0, height), rtol=rtol)


def test_normal_gravity_arrays():
    "Compare normal gravity passing arrays as arguments instead of floats"
    rtol = 1e-10
    heights = np.zeros(3)
    latitudes = np.array([-90, 90, 0])
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            gammas = np.array(
                [
                    get_ellipsoid().gravity_pole,
                    get_ellipsoid().gravity_pole,
                    get_ellipsoid().gravity_equator,
                ]
            )
            # Convert gammas to mGal
            gammas *= 1e5
            npt.assert_allclose(gammas, normal_gravity(latitudes, heights), rtol=rtol)


def test_normal_gravity_non_zero_height():
    "Normal gravity above and below the ellipsoid."
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            # Convert gamma to mGal
            gamma_pole = get_ellipsoid().gravity_pole * 1e5
            gamma_eq = get_ellipsoid().gravity_equator * 1e5
            assert gamma_pole > normal_gravity(90, 1000)
            assert gamma_pole > normal_gravity(-90, 1000)
            assert gamma_eq > normal_gravity(0, 1000)
            assert gamma_pole < normal_gravity(90, -1000)
            assert gamma_pole < normal_gravity(-90, -1000)
            assert gamma_eq < normal_gravity(0, -1000)


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
