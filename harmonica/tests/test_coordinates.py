"""
Test coordinates conversions.
"""
import numpy as np
import numpy.testing as npt

from ..coordinates import geodetic_to_geocentric
from ..ellipsoid import (
    KNOWN_ELLIPSOIDS,
    set_ellipsoid,
    get_ellipsoid,
    ReferenceEllipsoid,
)


def test_geodetic_geocentric_spherical_ellipsoid():
    "Test geodetic to geocentric coordinates conversion if ellipsoid is a sphere."
    rtol = 1e-10
    sphere_radius = 1.0
    # Define a "spherical" ellipsoid with radius equal to 1m.
    # To do so, we define a zero flattening, thus an infinite inverse flattening.
    spherical_ellipsoid = ReferenceEllipsoid(
        "unit_sphere", sphere_radius, np.infty, 0, 0
    )
    with set_ellipsoid(spherical_ellipsoid):
        latitude = np.linspace(-90, 90, 5)
        height = np.linspace(-0.2, 0.2, 5)
        geocentric_latitude, radius = geodetic_to_geocentric(latitude, height)
        npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
        npt.assert_allclose(radius, sphere_radius + height, rtol=rtol)


def test_geodetic_geocentric_on_equator():
    "Test geodetic to geocentric coordinates conversion on equator."
    rtol = 1e-10
    height = np.linspace(-1e4, 1e4, 5)
    latitude = np.full(height.shape, 0)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            geocentric_latitude, radius = geodetic_to_geocentric(latitude, height)
            npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semimajor_axis, rtol=rtol)


def test_geodetic_geocentric_on_poles():
    "Test geodetic to geocentric coordinates conversion on poles."
    rtol = 1e-10
    height = np.hstack([np.linspace(-1e4, 1e4, 5)] * 2)
    latitude = np.array([90.] * 5 + [-90.] * 5)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            geocentric_latitude, radius = geodetic_to_geocentric(latitude, height)
            npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semiminor_axis, rtol=rtol)
