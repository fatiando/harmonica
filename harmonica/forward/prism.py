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
    easting, northing, vertical = (i.ravel() for i in coordinates[:3])
    west, east, south, north, top, bottom = prism
    # First thing to do is make the computation point the origin of the coordinate
    # system
    x = [north - northing, south - northing]
    y = [east - easting, west - easting]
    z = [bottom - vertical, top - vertical]
    # Initialize with zeros
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros((len(components), cast.size), dtype=dtype)
    # Evaluate the integration limits
    for i, j, k in itertools.product(range(2), range(2), range(2)):
        prism_kernels(x[i], y[j], z[k], components, (-1)**(i + j + k),  result)
    # Now all that is left is to multiply result by the gravitational constant and
    # convert it to mGal units
    result *= GRAVITATIONAL_CONST*1e5*density
    result = [comp.reshape(cast.shape) for comp in result]
    if len(components) == 1:
        return result[0]
    return result


# @jit(nopython=True)
@vectorize
def atan2(y, x):
    """
    Correct the value of the angle returned by arctan2 to match the sign of the
    tangent. Also return 0 instead of 2Pi for 0 tangent.
    """
    result = math.atan2(y, x)
    if y > 0 and x < 0:
        result -= np.pi
    elif y < 0 and x < 0:
        result += np.pi
    return result


# @jit(nopython=True)
@vectorize
def log(x):
    """
    Return 0 for log(0) because the limits in the formula terms tend to 0
    (see Nagy et al., 2000)
    """
    if math.abs(x) < 1e-10:
        result = 0
    else:
        result = math.log(x)
    return result


@jit(nopython=True, fastmath=True)
def prism_kernels(x, y, z, components, scale, output):
    for i in range(output.shape[1]):
        r = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        for j in range(output.shape[0]):
            if components[j] == "v":
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential and gravity is -grad(V)
                kernel = -(x[i]*log(y[i] + r) +
                           y[i]*log(x[i] + r) -
                           z[i]*atan2(x[i]*y[i], z[i]*r))
            output[j, i] += scale*kernel
    return output

def kernel_nn(x, y, z, r):
    kernel = -safe_atan2(z*y, x*r)
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
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(z[k]*x[i], y[j]*r)
                res += ((-1.)**(i + j + k))*kernel
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
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(y[j]*x[i], z[k]*r)
                res += ((-1.)**(i + j + k))*kernel
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
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(z[k] + r)
                res += ((-1.)**(i + j + k))*kernel
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
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(y[j] + r)
                res += ((-1.)**(i + j + k))*kernel
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
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(x[i] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res
