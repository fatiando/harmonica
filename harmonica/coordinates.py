"""
Geographic coordinate conversion.
"""
import numpy as np

from . import get_ellipsoid


def geodetic_to_geocentric(latitude, height):
    """
    Convert from geodetic to geocentric coordinates.

    Parameters
    ----------
    latitude : float or array
        The geodetic latitude (in degrees).
    height : float or array
        The ellipsoidal (geometric or geodetic) height.

    Returns
    -------
    """
    ellipsoid = get_ellipsoid()
    # Convert latitude to radians
    latitude_rad = np.pi / 180 * latitude
    prime_vertical_radius = ellipsoid.semimajor_axis / np.sqrt(
        1 - ellipsoid.linear_eccentricity ** 2 * np.sin(latitude_rad) ** 2
    )
    xy_projection = (height + prime_vertical_radius) * np.cos(latitude_rad)
    z_cartesian = (
        height + (1 - ellipsoid.linear_eccentricity ** 2) * prime_vertical_radius
    ) * np.sin(latitude_rad)
    radius = np.sqrt(xy_projection ** 2 + z_cartesian ** 2)
    geocentric_latitude = 180 / np.pi * np.arcsin(z_cartesian / radius)
    return geocentric_latitude, radius
