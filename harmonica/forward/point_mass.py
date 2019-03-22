"""
Forward modelling for point masses
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST
from ..coordinates import geodetic_to_spherical


def point_mass_gravity(
    coordinates, point_mass, mass, field, coordinate_system="geodetic", dtype="float64"
):
    """
    Parameters
    ----------
    coordinates : list or array
        List or array containing `longitude`, `latitude` and `height` of computation
        points.
    point_mass : list or array
        Coordinates of the point mass: [`longitude`, `latitude`, `height`].
    """
    kernels = {"potential": kernel_potential, "gz": kernel_gz}
    if field not in kernels:
        raise ValueError("Gravity field {} not recognized".format(field))
    if coordinate_system not in ["geodetic", "spherical"]:
        raise ValueError(
            "Coordinate system {} not recognized".format(coordinate_system)
        )
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    if coordinate_system == "geodetic":
        longitude, latitude, height = (i.ravel() for i in coordinates[:3])
        longitude_p, latitude_p, height_p = point_mass[:]
        # Convert coordinates to geocentric spherical
        latitude, radius = geodetic_to_spherical(latitude, height)
        latitude_p, radius_p = geodetic_to_spherical(latitude_p, height_p)
        point_mass = [longitude_p, latitude_p, radius_p]
    elif coordinate_system == "spherical":
        longitude, latitude, radius = (i.ravel() for i in coordinates[:3])
    jit_point_mass_gravity(
        longitude, latitude, radius, point_mass, kernels[field], result
    )
    result *= GRAVITATIONAL_CONST * mass
    # Convert to more convenient units
    if field in ("gx", "gy", "gz"):
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


@jit(nopython=True)
def jit_point_mass_gravity(longitude, latitude, radius, point_mass, kernel, out):
    """
    """
    longitude_p, latitude_p, radius_p = point_mass[:]
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    radius_p_sq = radius_p ** 2
    for l in range(out.size):
        out[l] += kernel(
            longitude[l],
            latitude[l],
            radius[l],
            longitude_p,
            cosphi_p,
            sinphi_p,
            radius_p,
            radius_p_sq,
        )


@jit(nopython=True)
def kernel_potential(
    longitude, latitude, radius, longitude_p, cosphi_p, sinphi_p, radius_p, radius_p_sq
):
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance_sq = radius ** 2 + radius_p_sq - 2 * radius * radius_p * cospsi
    return 1 / np.sqrt(distance_sq)


@jit(nopython=True)
def kernel_gz(
    longitude, latitude, radius, longitude_p, cosphi_p, sinphi_p, radius_p, radius_p_sq
):
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance_sq = radius ** 2 + radius_p_sq - 2 * radius * radius_p * cospsi
    delta_z = radius_p * cospsi - radius
    return delta_z / distance_sq ** (3 / 2)
