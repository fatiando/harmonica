"""
Geographic coordinate conversion.
"""
import numpy as np

from . import get_ellipsoid


def geodetic_to_geocentric(latitude, height):
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
        The geocentric latitude (in degrees).
    radius : float or array
        The radial distance in the geocentric coordinate system (in meters), i.e.  the
        distance between the point given by ``latitude`` and ``height`` and the origin
        of the geocentric reference system.
    """
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    latitude_rad = np.pi / 180 * latitude
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
