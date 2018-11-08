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
    spherical_ellipsoid = ReferenceEllipsoid(
        "unit_sphere",
        sphere_radius,
        np.infty,
        0,
        0
    )
    with set_ellipsoid(spherical_ellipsoid):
        latitude = np.linspace(-90, 90, 5)
        height = np.linspace(-0.2, 0.2, 5)
        geocentric_latitude, radius = geodetic_to_geocentric(latitude, height)
        npt.assert_allclose(geocentric_latitude, latitude, rtol=rtol)
        npt.assert_allclose(radius, sphere_radius + height, rtol=rtol)
