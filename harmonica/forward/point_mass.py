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
        Coordinates of the point mass: [`longitude`, `latitude`, `height`].
    """
    kernels = {"potential": kernel_potential, "gz": kernel_gz}
    if field not in kernels:
        raise ValueError("Gravity field {} not recognized".format(field))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
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
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    radius_p_sq = radius_p ** 2
    for l in range(out.size):
        longitude_radians = np.radians(longitude[l])
        latitude_radians = np.radians(latitude[l])
        out[l] += kernel(
            longitude_radians,
            latitude_radians,
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
def kernel_gx(
    longitude, latitude, radius, longitude_p, cosphi_p, sinphi_p, radius_p, radius_p_sq
):
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance_sq = radius ** 2 + radius_p_sq - 2 * radius * radius_p * cospsi
    delta_x = radius_p * (cosphi * sinphi_p - sinphi * cosphi_p * coslambda)
    return delta_x / distance_sq ** (3 / 2)


@jit(nopython=True)
def kernel_gy(
    longitude, latitude, radius, longitude_p, cosphi_p, sinphi_p, radius_p, radius_p_sq
):
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    sinlambda = np.sin(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance_sq = radius ** 2 + radius_p_sq - 2 * radius * radius_p * cospsi
    delta_y = radius_p * cosphi_p * sinlambda
    return delta_y / distance_sq ** (3 / 2)


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
