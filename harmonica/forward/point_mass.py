"""
Forward modelling for point masses
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST
from ..coordinates import geodetic_to_spherical


def point_mass_gravity(coordinates, point_mass, mass, field, dtype="float64"):
    """
    Parameters
    ----------
    coordinates : list or array
        List or array containing `longitude`, `latitude` and `height` of computation
        points.
    point_mass : list or array
        Geodetic coordinates of the point mass: [`longitude`, `latitude`, `height`].
    """
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates of computation points from geodetic to geocentric spherical
    longitude, latitude, height = (i.ravel() for i in coordinates[:3])
    spherical_latitude, radius = geodetic_to_spherical(latitude, height)
    # Convert coordinates of point mass from geodetic to geocentric spherical
    mass_longitude, mass_latitude, mass_height = point_mass[:]
    mass_spherical_latitude, mass_radius = geodetic_to_spherical(
        mass_latitude, mass_height
    )
    point_mass_geocentric = [mass_longitude, mass_spherical_latitude, mass_radius]
    # Define kernels available and compute gravitational effect
    jit_point_mass_gravity(
        longitude,
        spherical_latitude,
        radius,
        point_mass_geocentric,
        field,
        result
    )
    result *= GRAVITATIONAL_CONST * mass
    # Convert to more convenient units
    if field in ("gx", "gy", "gz"):
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


@jit(nopython=True, fastmath=True)
def jit_point_mass_gravity(longitude, latitude, radius, point_mass, field, out):
    """
    """
    mass_longitude, mass_latitude, mass_radius = point_mass[:]
    cosphi_p = np.cos(mass_latitude)
    sinphi_p = np.sin(mass_latitude)
    mass_radius_sq = mass_radius ** 2
    for l in range(out.size):
        cosphi = (
            sinphi_p * np.sin(latitude[l])
            + cosphi_p * np.cos(latitude[l]) * np.cos(mass_longitude - longitude[l])
        )
        distance_sq = radius**2 + mass_radius_sq - 2 * radius * mass_radius * cosphi
        if field == "potential":
            out[l] += 1 / np.sqrt(distance_sq)
