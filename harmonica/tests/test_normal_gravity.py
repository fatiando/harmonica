"""
Testing normal gravity calculation.
"""
import numpy as np
import numpy.testing as npt

from .. import normal_gravity
from .. import set_ellipsoid, get_ellipsoid
from ..ellipsoid import KNOWN_ELLIPSOIDS


def test_normal_gravity():
    "Compare normal gravity values at pole and equator"
    rtol = 1e-10
    height = 0
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            # Convert gamma to mGal
            gamma_pole = get_ellipsoid().gravity_pole * 1e-5
            gamma_eq = get_ellipsoid().gravity_equator * 1e-5
            npt.assert_allclose(gamma_pole, normal_gravity(-90, height),
                                rtol=rtol)
            npt.assert_allclose(gamma_pole, normal_gravity(90, height),
                                rtol=rtol)
            npt.assert_allclose(gamma_eq, normal_gravity(0, height), rtol=rtol)


def test_normal_gravity_arrays():
    "Compare normal gravity passing arrays as arguments instead of floats"
    rtol = 1e-10
    heights = np.zeros(3)
    latitudes = np.array([-90, 90, 0])
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            gammas = np.array([get_ellipsoid().gravity_pole,
                               get_ellipsoid().gravity_pole,
                               get_ellipsoid().gravity_equator])
            # Convert gammas to mGal
            gammas *= 1e-5
            npt.assert_allclose(gammas, normal_gravity(latitudes, heights),
                                rtol=rtol)


def test_no_zero_height():
    "Normal gravity above and below the ellipsoid."
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            # Convert gamma to mGal
            gamma_pole = get_ellipsoid().gravity_pole * 1e-5
            gamma_eq = get_ellipsoid().gravity_equator * 1e-5
            assert gamma_pole > normal_gravity(90, 1000)
            assert gamma_pole > normal_gravity(-90, 1000)
            assert gamma_eq > normal_gravity(0, 1000)
            assert gamma_pole < normal_gravity(90, -1000)
            assert gamma_pole < normal_gravity(-90, -1000)
            assert gamma_eq < normal_gravity(0, -1000)
