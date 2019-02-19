"""
Test coordinates conversions.
"""
import numpy as np
import numpy.testing as npt

from ..coordinates import geodetic_to_spherical, spherical_to_geodetic
from ..ellipsoid import (
    KNOWN_ELLIPSOIDS,
    set_ellipsoid,
    get_ellipsoid,
    ReferenceEllipsoid,
)


def test_geodetic_to_spherical_with_spherical_ellipsoid():
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
        geocentric_latitude, radius = geodetic_to_spherical(latitude, height)
        npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
        npt.assert_allclose(radius, sphere_radius + height, rtol=rtol)


def test_geodetic_to_spherical_on_equator():
    "Test geodetic to geocentric coordinates conversion on equator."
    rtol = 1e-10
    height = np.linspace(-1e4, 1e4, 5)
    latitude = np.full(height.shape, 0)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            geocentric_latitude, radius = geodetic_to_spherical(latitude, height)
            npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semimajor_axis, rtol=rtol)


def test_geodetic_to_spherical_on_poles():
    "Test geodetic to geocentric coordinates conversion on poles."
    rtol = 1e-10
    height = np.hstack([np.linspace(-1e4, 1e4, 5)] * 2)
    latitude = np.array([90.0] * 5 + [-90.0] * 5)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            geocentric_latitude, radius = geodetic_to_spherical(latitude, height)
            npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semiminor_axis, rtol=rtol)


def test_spherical_to_geodetic_with_spherical_ellipsoid():
    "Test spherical to geodetic coordinates conversion if ellipsoid is a sphere."
    rtol = 1e-10
    sphere_radius = 1.0
    # Define a "spherical" ellipsoid with radius equal to 1m.
    # To do so, we define a zero flattening, thus an infinite inverse flattening.
    spherical_ellipsoid = ReferenceEllipsoid(
        "unit_sphere", sphere_radius, np.infty, 0, 0
    )
    with set_ellipsoid(spherical_ellipsoid):
        geocentric_latitude = np.linspace(-90, 90, 5)
        radius = np.linspace(0.8, 1.2, 5)
        latitude, height = spherical_to_geodetic(geocentric_latitude, radius)
        npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
        npt.assert_allclose(radius, sphere_radius + height, rtol=rtol)


def test_spherical_to_geodetic_on_equator():
    "Test spherical to geodetic coordinates conversion on equator."
    rtol = 1e-10
    size = 5
    geocentric_latitude = np.zeros(size)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            radius = np.linspace(-1e4, 1e4, size) + ellipsoid.semimajor_axis
            latitude, height = spherical_to_geodetic(geocentric_latitude, radius)
            npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semimajor_axis, rtol=rtol)


def test_spherical_to_geodetic_on_poles():
    "Test spherical to geodetic coordinates conversion on poles."
    rtol = 1e-10
    size = 5
    geocentric_latitude = np.array([90.0] * size + [-90.0] * size)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            radius = np.hstack(
                [np.linspace(-1e4, 1e4, size) + ellipsoid.semiminor_axis] * 2
            )
            latitude, height = spherical_to_geodetic(geocentric_latitude, radius)
            npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semiminor_axis, rtol=rtol)


def test_identity():
    "Test if geodetic_to_spherical and spherical_to_geodetic is the identity operator"
    rtol = 1e-10
    latitude = np.linspace(-90, 90, 19)
    height = np.linspace(-1e4, 1e4, 8)
    latitude, height = np.meshgrid(latitude, height)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            geocentric_latitude, radius = geodetic_to_spherical(latitude, height)
            converted_latitude, converted_height = spherical_to_geodetic(
                geocentric_latitude, radius
            )
            npt.assert_allclose(latitude, converted_latitude, rtol=rtol)
            npt.assert_allclose(height, converted_height, rtol=rtol)
