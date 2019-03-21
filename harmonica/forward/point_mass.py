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
    """
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates of computation points from geodetic to geocentric spherical
    longitude, latitude, height = (i.ravel() for i in coordinates[:3])
    latitude, height = geodetic_to_spherical(latitude, height)
    kernels = {"potential": kernel_potential, "gz": kernel_z}
    jit_point_mass_gravity(
        longitude, latitude, height, point_mass, kernels[field], result
    )
    result *= GRAVITATIONAL_CONST * mass
    # Convert to more convenient units
    if field in ("gx", "gy", "gz"):
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)
