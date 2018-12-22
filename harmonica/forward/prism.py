"""
Forward modeling for right-rectangular prisms.
"""
import itertools
import math

import numpy as np
from numba import jit, vectorize

from ..constants import GRAVITATIONAL_CONST


def prism_gravity(coordinates, prism, density, components, dtype="float64"):
    """
    """
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    easting, northing, vertical = (i.ravel() for i in coordinates[:3])
    result = {comp: np.zeros(cast.size, dtype=dtype) for comp in components}
    kernels = dict(v=kernel_v)
    for comp in components:
        _prism_gravity(easting, northing, vertical, prism, kernels[comp], result[comp])
    # Now all that is left is to multiply result by the gravitational constant and
    # convert it to mGal units
    for comp in components:
        result[comp] *= GRAVITATIONAL_CONST * 1e5 * density
        result[comp] = result[comp].reshape(cast.shape)
    if len(components) == 1:
        return result[components[0]]
    return result


@jit(nopython=True, fastmath=True)
def _prism_gravity(easting, northing, vertical, prism, kernel, out):
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
def kernel_v(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    kernel = x * log(y + r) + y * log(x + r) - z * atan2(x * y, z * r)
    # Minus because Nagy et al (2000) give the formula for the
    # gradient of the potential and gravity is -grad(V)
    return -kernel


@jit(nopython=True)
def atan2(y, x):
    """
    Correct the value of the angle returned by arctan2 to match the sign of the
    tangent. Also return 0 instead of 2Pi for 0 tangent.
    """
    result = np.arctan2(y, x)
    if y > 0 and x < 0:
        result -= np.pi
    elif y < 0 and x < 0:
        result += np.pi
    return result


@jit(nopython=True)
def log(x):
    """
    Return 0 for log(0) because the limits in the formula terms tend to 0
    (see Nagy et al., 2000)
    """
    if np.abs(x) < 1e-10:
        result = 0
    else:
        result = np.log(x)
    return result


def kernel_nn(x, y, z, r):
    kernel = -safe_atan2(z * y, x * r)
    return kernel


def kernelyy(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                kernel = -safe_atan2(z[k] * x[i], y[j] * r)
                res += ((-1.0) ** (i + j + k)) * kernel
    return res


def kernelzz(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                kernel = -safe_atan2(y[j] * x[i], z[k] * r)
                res += ((-1.0) ** (i + j + k)) * kernel
    return res


def kernelxy(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                kernel = safe_log(z[k] + r)
                res += ((-1.0) ** (i + j + k)) * kernel
    return res


def kernelxz(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                kernel = safe_log(y[j] + r)
                res += ((-1.0) ** (i + j + k)) * kernel
    return res


def kernelyz(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                kernel = safe_log(x[i] + r)
                res += ((-1.0) ** (i + j + k)) * kernel
    return res
