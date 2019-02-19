"""
Geographic coordinate conversion.
"""
import numpy as np

from . import get_ellipsoid


def geodetic_to_spherical(latitude, height):
    """
    Convert from geodetic to geocentric spherical coordinates.

    The geodetic datum is defined by the default :class:`harmonica.ReferenceEllipsoid`
    set by the :func:`harmonica.set_ellipsoid` function.
    The coordinates are converted following [Vermeille2002]_.

    Parameters
    ----------
    latitude : float or array
        The geodetic latitude (in degrees).
    height : float or array
        The ellipsoidal (geometric or geodetic) height (in meters).

    Returns
    -------
    geocentric_latitude : float or array
        The latitude coordinate in the geocentric spherical reference system
        (in degrees).
    radius : float or array
        The radial coordinate in the geocentric spherical reference system (in meters).
    """
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    latitude_rad = np.radians(latitude)
    prime_vertical_radius = ellipsoid.semimajor_axis / np.sqrt(
        1 - ellipsoid.first_eccentricity ** 2 * np.sin(latitude_rad) ** 2
    )
    # Instead of computing X and Y, we only comupute the projection on the XY plane:
    # xy_projection = sqrt( X**2 + Y**2 )
    xy_projection = (height + prime_vertical_radius) * np.cos(latitude_rad)
    z_cartesian = (
        height + (1 - ellipsoid.first_eccentricity ** 2) * prime_vertical_radius
    ) * np.sin(latitude_rad)
    radius = np.sqrt(xy_projection ** 2 + z_cartesian ** 2)
    geocentric_latitude = 180 / np.pi * np.arcsin(z_cartesian / radius)
    return geocentric_latitude, radius


def spherical_to_geodetic(geocentric_latitude, radius):
    """
    Convert from geocentric spherical to geodetic coordinates.

    The geodetic datum is defined by the default :class:`harmonica.ReferenceEllipsoid`
    set by the :func:`harmonica.set_ellipsoid` function.
    The coordinates are converted following [Vermeille2002]_.

    Parameters
    ----------
    geocentric_latitude : float or array
        The latitude coordinate in the geocentric spherical reference system
        (in degrees).
    radius : float or array
        The radial coordinate in the geocentric spherical reference system (in meters).

    Returns
    -------
    latitude : float or array
        The geodetic latitude (in degrees).
    height : float or array
        The ellipsoidal (geometric or geodetic) height (in meters).
    """
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    geocentric_latitude_rad = np.radians(geocentric_latitude)
    Z = radius * np.sin(geocentric_latitude_rad)
    p = radius ** 2 * np.cos(geocentric_latitude_rad) ** 2 / \
        ellipsoid.semimajor_axis ** 2
    q = (1 - ellipsoid.first_eccentricity ** 2) / ellipsoid.semimajor_axis ** 2 * Z ** 2
    r = (p + q - ellipsoid.first_eccentricity ** 4) / 6
    s = ellipsoid.first_eccentricity ** 4 * p * q / 4 / r ** 3
    t = np.cbrt(1 + s + np.sqrt(2 * s + s ** 2))
    u = r * (1 + t + 1 / t)
    v = np.sqrt(u ** 2 + q * ellipsoid.first_eccentricity ** 4)
    w = ellipsoid.first_eccentricity ** 2 * (u + v - q) / 2 / v
    k = np.sqrt(u + v + w ** 2) - w
    D = k * radius * np.cos(geocentric_latitude_rad) / \
        (k + ellipsoid.first_eccentricity ** 2)
    latitude = np.degrees(2 * np.arctan(Z / (D + np.sqrt(D ** 2 + Z ** 2))))
    height = (k + ellipsoid.first_eccentricity ** 2 - 1) / k * np.sqrt(D ** 2 + Z ** 2)
    return latitude, height
