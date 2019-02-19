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

    See also
    --------
    spherical_to_geodetic : Convert from geocentric spherical to geodetic coordinates.
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
    geocentric_latitude = np.degrees(np.arcsin(z_cartesian / radius))
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

    See also
    --------
    geodetic_to_spherical : Convert from geodetic to geocentric spherical coordinates.
    """
    ellipsoid = get_ellipsoid()
    k, big_d, big_z = _spherical_to_geodetic_parameters(geocentric_latitude, radius)
    latitude = np.degrees(
        2 * np.arctan(big_z / (big_d + np.sqrt(big_d ** 2 + big_z ** 2)))
    )
    height = (
        (k + ellipsoid.first_eccentricity ** 2 - 1)
        / k
        * np.sqrt(big_d ** 2 + big_z ** 2)
    )
    return latitude, height


def _spherical_to_geodetic_parameters(geocentric_latitude, radius):
    "Compute parameters for spherical to geodetic coordinates conversion"
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    geocentric_latitude_rad = np.radians(geocentric_latitude)
    big_z = radius * np.sin(geocentric_latitude_rad)
    p_0 = (
        radius ** 2
        * np.cos(geocentric_latitude_rad) ** 2
        / ellipsoid.semimajor_axis ** 2
    )
    q_0 = (
        (1 - ellipsoid.first_eccentricity ** 2)
        / ellipsoid.semimajor_axis ** 2
        * big_z ** 2
    )
    r_0 = (p_0 + q_0 - ellipsoid.first_eccentricity ** 4) / 6
    s_0 = ellipsoid.first_eccentricity ** 4 * p_0 * q_0 / 4 / r_0 ** 3
    t_0 = np.cbrt(1 + s_0 + np.sqrt(2 * s_0 + s_0 ** 2))
    u_0 = r_0 * (1 + t_0 + 1 / t_0)
    v_0 = np.sqrt(u_0 ** 2 + q_0 * ellipsoid.first_eccentricity ** 4)
    w_0 = ellipsoid.first_eccentricity ** 2 * (u_0 + v_0 - q_0) / 2 / v_0
    k = np.sqrt(u_0 + v_0 + w_0 ** 2) - w_0
    big_d = (
        k
        * radius
        * np.cos(geocentric_latitude_rad)
        / (k + ellipsoid.first_eccentricity ** 2)
    )
    return k, big_d, big_z
