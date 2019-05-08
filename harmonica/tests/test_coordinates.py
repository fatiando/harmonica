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
    size = 5
    longitude = np.linspace(0, 180, size)
    with set_ellipsoid(spherical_ellipsoid):
        latitude = np.linspace(-90, 90, size)
        height = np.linspace(-0.2, 0.2, size)
        spherical_longitude, spherical_latitude, radius = geodetic_to_spherical(
            [longitude, latitude, height]
        )
        npt.assert_allclose(spherical_longitude, longitude, rtol=rtol)
        npt.assert_allclose(spherical_latitude, latitude, rtol=rtol)
        npt.assert_allclose(radius, sphere_radius + height, rtol=rtol)


def test_geodetic_to_spherical_on_equator():
    "Test geodetic to geocentric coordinates conversion on equator."
    rtol = 1e-10
    size = 5
    longitude = np.linspace(0, 180, size)
    height = np.linspace(-1e4, 1e4, size)
    latitude = np.zeros_like(size)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            spherical_longitude, spherical_latitude, radius = geodetic_to_spherical(
                [longitude, latitude, height]
            )
            npt.assert_allclose(spherical_longitude, longitude, rtol=rtol)
            npt.assert_allclose(spherical_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semimajor_axis, rtol=rtol)


def test_geodetic_to_spherical_on_poles():
    "Test geodetic to geocentric coordinates conversion on poles."
    rtol = 1e-10
    size = 5
    longitude = np.hstack([np.linspace(0, 180, size)] * 2)
    height = np.hstack([np.linspace(-1e4, 1e4, size)] * 2)
    latitude = np.array([90.0] * size + [-90.0] * size)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            spherical_longitude, spherical_latitude, radius = geodetic_to_spherical(
                [longitude, latitude, height]
            )
            npt.assert_allclose(spherical_longitude, longitude, rtol=rtol)
            npt.assert_allclose(spherical_latitude, latitude, rtol=rtol)
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
    size = 5
    spherical_longitude = np.linspace(0, 180, size)
    with set_ellipsoid(spherical_ellipsoid):
        spherical_latitude = np.linspace(-90, 90, size)
        radius = np.linspace(0.8, 1.2, size)
        longitude, latitude, height = spherical_to_geodetic(
            [spherical_longitude, spherical_latitude, radius]
        )
        npt.assert_allclose(spherical_longitude, longitude, rtol=rtol)
        npt.assert_allclose(spherical_latitude, latitude, rtol=rtol)
        npt.assert_allclose(radius, sphere_radius + height, rtol=rtol)


def test_spherical_to_geodetic_on_equator():
    "Test spherical to geodetic coordinates conversion on equator."
    rtol = 1e-10
    size = 5
    spherical_latitude = np.zeros(size)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            spherical_longitude = np.linspace(0, 180, size)
            radius = np.linspace(-1e4, 1e4, size) + ellipsoid.semimajor_axis
            longitude, latitude, height = spherical_to_geodetic(
                [spherical_longitude, spherical_latitude, radius]
            )
            npt.assert_allclose(spherical_longitude, longitude, rtol=rtol)
            npt.assert_allclose(spherical_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semimajor_axis, rtol=rtol)


def test_spherical_to_geodetic_on_poles():
    "Test spherical to geodetic coordinates conversion on poles."
    rtol = 1e-10
    size = 5
    spherical_longitude = np.hstack([np.linspace(0, 180, size)] * 2)
    spherical_latitude = np.array([90.0] * size + [-90.0] * size)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            ellipsoid = get_ellipsoid()
            radius = np.hstack(
                [np.linspace(-1e4, 1e4, size) + ellipsoid.semiminor_axis] * 2
            )
            longitude, latitude, height = spherical_to_geodetic(
                [spherical_longitude, spherical_latitude, radius]
            )
            npt.assert_allclose(spherical_longitude, longitude, rtol=rtol)
            npt.assert_allclose(spherical_latitude, latitude, rtol=rtol)
            npt.assert_allclose(radius, height + ellipsoid.semiminor_axis, rtol=rtol)


def test_identity():
    "Test if geodetic_to_spherical and spherical_to_geodetic is the identity operator"
    rtol = 1e-10
    longitude = np.linspace(0, 350, 36)
    latitude = np.linspace(-90, 90, 19)
    height = np.linspace(-1e4, 1e4, 8)
    coordinates = np.meshgrid(longitude, latitude, height)
    for ellipsoid_name in KNOWN_ELLIPSOIDS:
        with set_ellipsoid(ellipsoid_name):
            spherical_coordinates = geodetic_to_spherical(coordinates)
            reconverted_coordinates = spherical_to_geodetic(spherical_coordinates)
            npt.assert_allclose(coordinates, reconverted_coordinates, rtol=rtol)
