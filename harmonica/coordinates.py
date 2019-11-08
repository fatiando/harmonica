"""
Geographic coordinate conversion.
"""
import numpy as np

from . import get_ellipsoid


def geodetic_to_spherical(longitude, latitude, height):
    """
    Convert from geodetic to geocentric spherical coordinates.

    The geodetic datum is defined by the default
    :class:`harmonica.ReferenceEllipsoid` set by the
    :func:`harmonica.set_ellipsoid` function.
    The coordinates are converted following [Vermeille2002]_.

    Parameters
    ----------
    longitude : array
        Longitude coordinates on geodetic coordinate system in degrees.
    latitude : array
        Latitude coordinates on geodetic coordinate system in degrees.
    height : array
        Ellipsoidal heights in meters.

    Returns
    -------
    longitude : array
        Longitude coordinates on geocentric spherical coordinate system in
        degrees.
        The longitude coordinates are not modified during this conversion.
    spherical_latitude : array
        Converted latitude coordinates on geocentric spherical coordinate
        system in degrees.
    radius : array
        Converted spherical radius coordinates in meters.

    See also
    --------
    spherical_to_geodetic :
        Convert from geocentric spherical to geodetic coordinates.

    Examples
    --------

    In the poles, the radius should be the reference ellipsoid's semi-minor
    axis:

    >>> import harmonica as hm
    >>> spherical = hm.geodetic_to_spherical(
    ...     longitude=0, latitude=90, height=0
    ... )
    >>> print(", ".join("{:.4f}".format(i) for i in spherical))
    0.0000, 90.0000, 6356752.3142
    >>> print("{:.4f}".format(hm.get_ellipsoid().semiminor_axis))
    6356752.3142

    In  the equator, it should be the semi-major axis:

    >>> spherical = hm.geodetic_to_spherical(longitude=0, latitude=0, height=0)
    >>> print(", ".join("{:.4f}".format(i) for i in spherical))
    0.0000, 0.0000, 6378137.0000
    >>> print("{:.4f}".format(hm.get_ellipsoid().semimajor_axis))
    6378137.0000

    """
    # Get ellipsoid
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    latitude_rad = np.radians(latitude)
    prime_vertical_radius = ellipsoid.semimajor_axis / np.sqrt(
        1 - ellipsoid.first_eccentricity ** 2 * np.sin(latitude_rad) ** 2
    )
    # Instead of computing X and Y, we only comupute the projection on the XY
    # plane: xy_projection = sqrt( X**2 + Y**2 )
    xy_projection = (height + prime_vertical_radius) * np.cos(latitude_rad)
    z_cartesian = (
        height + (1 - ellipsoid.first_eccentricity ** 2) * prime_vertical_radius
    ) * np.sin(latitude_rad)
    radius = np.sqrt(xy_projection ** 2 + z_cartesian ** 2)
    spherical_latitude = np.degrees(np.arcsin(z_cartesian / radius))
    return longitude, spherical_latitude, radius


def spherical_to_geodetic(longitude, spherical_latitude, radius):
    """
    Convert from geocentric spherical to geodetic coordinates.

    The geodetic datum is defined by the default
    :class:`harmonica.ReferenceEllipsoid` set by the
    :func:`harmonica.set_ellipsoid` function.
    The coordinates are converted following [Vermeille2002]_.

    Parameters
    ----------
    longitude : array
        Longitude coordinates on geocentric spherical coordinate system in
        degrees.
    spherical_latitude : array
        Latitude coordinates on geocentric spherical coordinate system in
        degrees.
    radius : array
        Spherical radius coordinates in meters.

    Returns
    -------
    longitude : array
        Longitude coordinates on geodetic coordinate system in degrees.
        The longitude coordinates are not modified during this conversion.
    latitude : array
        Converted latitude coordinates on geodetic coordinate system in
        degrees.
    height : array
        Converted ellipsoidal height coordinates in meters.

    See also
    --------
    geodetic_to_spherical :
        Convert from geodetic to geocentric spherical coordinates.

    Examples
    --------

    In the poles and equator, using the semi-minor or semi-major axis of the
    ellipsoid as the radius should yield 0 height:

    >>> import harmonica as hm
    >>> geodetic = hm.spherical_to_geodetic(
    ...     longitude=0,
    ...     spherical_latitude=90,
    ...     radius=hm.get_ellipsoid().semiminor_axis,
    ... )
    >>> print(", ".join("{:.1f}".format(i) for i in geodetic))
    0.0, 90.0, 0.0
    >>> geodetic = hm.spherical_to_geodetic(
    ...     longitude=0,
    ...     spherical_latitude=0,
    ...     radius=hm.get_ellipsoid().semimajor_axis,
    ... )
    >>> print(", ".join("{:.1f}".format(i) for i in geodetic))
    0.0, 0.0, 0.0
    >>> geodetic = hm.spherical_to_geodetic(
    ...     longitude=0,
    ...     spherical_latitude=-90,
    ...     radius=hm.get_ellipsoid().semiminor_axis + 2
    ... )
    >>> print(", ".join("{:.1f}".format(i) for i in geodetic))
    0.0, -90.0, 2.0

    """
    # Get ellipsoid
    ellipsoid = get_ellipsoid()
    k, big_d, big_z = _spherical_to_geodetic_parameters(spherical_latitude, radius)
    latitude = np.degrees(
        2 * np.arctan(big_z / (big_d + np.sqrt(big_d ** 2 + big_z ** 2)))
    )
    height = (
        (k + ellipsoid.first_eccentricity ** 2 - 1)
        / k
        * np.sqrt(big_d ** 2 + big_z ** 2)
    )
    return longitude, latitude, height


def _spherical_to_geodetic_parameters(spherical_latitude, radius):
    "Compute parameters for spherical to geodetic coordinates conversion"
    # Get ellipsoid
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    spherical_latitude_rad = np.radians(spherical_latitude)
    big_z = radius * np.sin(spherical_latitude_rad)
    p_0 = (
        radius ** 2
        * np.cos(spherical_latitude_rad) ** 2
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
        * np.cos(spherical_latitude_rad)
        / (k + ellipsoid.first_eccentricity ** 2)
    )
    return k, big_d, big_z
