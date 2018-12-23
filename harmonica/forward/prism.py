"""
Forward modeling for right-rectangular prisms.
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST


def prism_gravity(coordinates, prism, density, field, dtype="float64"):
    """
    """
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    easting, northing, vertical = (i.ravel() for i in coordinates[:3])
    kernels = {"potential": kernel_potential, "gz": kernel_z}
    result = np.zeros(cast.size, dtype=dtype)
    jit_prism_gravity(easting, northing, vertical, prism, kernels[field], result)
    result *= GRAVITATIONAL_CONST * density
    # Convert to more convenient units
    if field in ("gx", "gy", "gz"):
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


@jit(nopython=True, fastmath=True)
def jit_prism_gravity(easting, northing, vertical, prism, kernel, out):
    """
    """
    for l in range(out.size):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    shift_east = prism[1 - i]
                    shift_north = prism[3 - j]
                    shift_down = prism[5 - k]
                    out[l] += (-1) ** (i + j + k) * kernel(
                        shift_east - easting[l],
                        shift_north - northing[l],
                        shift_down - vertical[l],
                    )


@jit(nopython=True)
def kernel_z(east, north, vertical):
    """
    """
    radius = np.sqrt(east ** 2 + north ** 2 + vertical ** 2)
    kernel = (
        east * log(north + radius)
        + north * log(east + radius)
        - vertical * atan2(east * north, vertical * radius)
    )
    # Minus because Nagy et al (2000) give the formula for the
    # gradient of the potential and gravity is -grad(V)
    return -kernel


@jit(nopython=True)
def kernel_potential(east, north, vertical):
    """
    """
    radius = np.sqrt(east ** 2 + north ** 2 + vertical ** 2)
    kernel = (
        east * north * log(vertical + radius)
        + north * vertical * log(east + radius)
        + east * vertical * log(north + radius)
        - 0.5 * east ** 2 * atan2(vertical * north, east * radius)
        - 0.5 * north ** 2 * atan2(vertical * east, north * radius)
        - 0.5 * vertical ** 2 * atan2(east * north, vertical * radius)
    )
    return kernel


@jit(nopython=True)
def atan2(y, x):
    """
    Modified arctangent to return angles in the right quadrants for prism modeling.
    Needed to get the correct results below the prism and on the sides (for some
    components).
    Shift the angle from np.arctan2 to match the sign of the tangent.
    Return 0 instead of 2Pi for 0 tangent.
    """
    if np.abs(y) < 1e-10:
        result = 0
    else:
        result = np.arctan2(y, x)
    if y > 0 and x < 0:
        result -= np.pi
    elif y < 0 and x < 0:
        result += np.pi
    return result


@jit(nopython=True)
def log(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0 (see Nagy et al., 2000)
    """
    if np.abs(x) < 1e-10:
        result = 0
    else:
        result = np.log(x)
    return result
