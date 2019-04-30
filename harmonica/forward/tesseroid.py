"""
Forward modelling for tesseroids
"""
import numpy as np
from numba import jit
from numpy.polynomial.legendre import leggauss

from ..constants import GRAVITATIONAL_CONST
from .point_mass import (
    jit_point_mass_gravity,
    kernel_potential,
    kernel_gx,
    kernel_gy,
    kernel_gz,
    kernel_gxx,
    kernel_gxy,
    kernel_gxz,
    kernel_gyy,
    kernel_gyz,
    kernel_gzz,
)

STACK_SIZE = 100
GLQ_DEGREES = [2, 2, 2]
DISTANCE_SIZE_RATIO_POTENTIAL = 1
DISTANCE_SIZE_RATIO_ACCELERATION = 2.5
DISTANCE_SIZE_RATIO_TENSOR = 8


def tesseroid_gravity(coordinates, tesseroid, density, field):
    """
    Compute gravitational field of a tesseroid on a single computation point

    Parameters
    ----------
    coordinates: list or 1d-array
        List or array containing `longitude`, `latitude` and `radius` of a single
        computation points defined on a spherical geocentric coordinate system.
        Both `longitude` and `latitude` should be in degrees and `radius` in meters.
    tesseroid : list or 1d-array
        Geocentric spherical coordinates of the tesseroid: `w`, `e`, `s`, `n`, `bottom`,
        `top`.
        The longitudinal and latitudinal boundaries should be in degrees, while the
        radial ones must be in meters.
    density : float
        Density of the single tesseroid in kg/m^3.
    field: str
        Gravitational field that wants to be computed.
        The available fields are:

        - Gravitational potential: ``potential``
        - Accelerations or gradient components: ``gx``, ``gy``, ``gz``
        - Maurssi tensor components: ``gxx``, ``gxy``, ``gxz``, ``gyy``, ``gyz``,
          ``gzz``
    """
    kernels = {
        "potential": kernel_potential,
        "gx": kernel_gx,
        "gy": kernel_gy,
        "gz": kernel_gz,
        "gxx": kernel_gxx,
        "gxy": kernel_gxy,
        "gxz": kernel_gxz,
        "gyy": kernel_gyy,
        "gyz": kernel_gyz,
        "gzz": kernel_gzz,
    }
    if field not in kernels:
        raise ValueError("Gravity field {} not recognized".format(field))
    # Get value of D (distance_size_ratio)
    if field == "potential":
        distance_size_ratio = DISTANCE_SIZE_RATIO_POTENTIAL
    elif field in ("gx", "gy", "gz"):
        distance_size_ratio = DISTANCE_SIZE_RATIO_ACCELERATION
    elif field in ("gxx", "gxy", "gxz", "gyy", "gyz", "gzz"):
        distance_size_ratio = DISTANCE_SIZE_RATIO_TENSOR
    # Apply adaptive discretization on tesseroid
    small_tesseroids = adaptive_discretization(
        coordinates, tesseroid, distance_size_ratio
    )
    # Initialize result
    result = 0
    jit_point_mass_gravity(
        longitude, latitude, radius, point_mass, kernels[field], result
    )


def tesseroid_to_point_masses(tesseroid, glq_degrees=GLQ_DEGREES):
    """
    Convert tesseroid to equivalent point masses on nodes of GLQ
    """
    # Get nodes coordinates and weights
    lon_degree, lat_degree, rad_degree = glq_degrees[:]
    lon_node, lon_weights = leggauss(lon_degree)
    lat_node, lat_weights = leggauss(lat_degree)
    rad_node, rad_weights = leggauss(rad_degree)
    # Get coordinates of the tesseroid
    w, e, s, n, bottom, top = tesseroid[:]
    # Scale nodes
    longitude = 0.5 * (e - w) * lon_node + 0.5 * (e + w)
    latitude = 0.5 * (n - s) * lat_node + 0.5 * (n + s)
    radius = 0.5 * (top - bottom) * rad_node + 0.5 * (top + bottom)
    # Create mesh grids of coordinates and weights
    longitude, latitude, radius = np.meshgrid(longitude, latitude, radius)
    lon_weights, lat_weights, rad_weights = np.meshgrid(
        lon_weights, lat_weights, rad_weights
    )
    point_masses = [longitude, latitude, radius]
    weights = [lon_weights, lat_weights, rad_weights]
    return (point_masses, weights)


@jit(nopython=True)
def adaptive_discretization(
    coordinates,
    tesseroid,
    distance_size_ratio,
    stack_size=STACK_SIZE,
    radial_discretization=False,
):
    """
    Three or two dimensional adaptive discretization
    """
    # Create list of small tesseroids
    small_tesseroids = []
    # Create stack of tesseroids
    stack = np.zeros((stack_size, 6))
    stack[0, :] = tesseroid
    stack_top = 0
    while stack_top >= 0:
        # Pop the first tesseroid from the stack
        tesseroid = [stack[stack_top, i] for i in range(6)]
        stack_top -= 1
        # Get its dimensions
        l_lon, l_lat, l_rad = _tesseroid_dimensions(tesseroid)
        # Get distance between computation point and center of tesseroid
        distance = _distance_tesseroid_point(coordinates, tesseroid)
        # Check inequality
        n_lon, n_lat, n_rad = 1, 1, 1
        if distance / l_lon < distance_size_ratio:
            n_lon = 2
        if distance / l_lat < distance_size_ratio:
            n_lat = 2
        if distance / l_rad < distance_size_ratio and radial_discretization:
            n_rad = 2
        # Apply discretization
        if n_lon * n_lat * n_rad > 1:
            stack_top = _split_tesseroid(
                tesseroid, n_lon, n_lat, n_rad, stack, stack_top
            )
        else:
            small_tesseroids.append(tesseroid)
    return np.array(small_tesseroids)


@jit(nopython=True)
def _split_tesseroid(tesseroid, n_lon, n_lat, n_rad, stack, stack_top):
    """
    Split tesseroid along each dimension
    """
    w, e, s, n, bottom, top = tesseroid[:]
    if stack_top + n_lon * n_lat * n_rad > stack.shape[0]:
        raise OverflowError("Tesseroid stack overflow.")
    # Compute differential distance
    d_lon = (e - w) / n_lon
    d_lat = (n - s) / n_lat
    d_rad = (top - bottom) / n_rad
    for i in range(n_lon):
        for j in range(n_lat):
            for k in range(n_rad):
                stack_top += 1
                stack[stack_top, 0] = w + d_lon * i
                stack[stack_top, 1] = w + d_lon * (i + 1)
                stack[stack_top, 2] = s + d_lat * j
                stack[stack_top, 3] = s + d_lat * (j + 1)
                stack[stack_top, 4] = bottom + d_rad * k
                stack[stack_top, 5] = bottom + d_rad * (k + 1)
    return stack_top


@jit(nopython=True)
def _tesseroid_dimensions(tesseroid):
    """
    Calculate the dimensions of the tesseroid.
    """
    w, e, s, n, bottom, top = tesseroid[:]
    w, e, s, n = np.radians(w), np.radians(e), np.radians(s), np.radians(n)
    latitude_center = (n + s) / 2
    l_lat = top * np.arccos(np.sin(n) * np.sin(s) + np.cos(n) * np.cos(s))
    l_lon = top * np.arccos(
        np.sin(latitude_center) ** 2 + np.cos(latitude_center) ** 2 * np.cos(e - w)
    )
    l_rad = top - bottom
    return l_lon, l_lat, l_rad


@jit(nopython=True)
def _distance_tesseroid_point(coordinates, tesseroid):
    """
    Calculate the distance between a computation point and the center of a tesseroid.
    """
    longitude, latitude, radius = coordinates[:]
    longitude, latitude = np.radians(longitude), np.radians(latitude)
    # Get center of the tesseroid
    longitude_p = (tesseroid[0] + tesseroid[1]) / 2
    latitude_p = (tesseroid[2] + tesseroid[3]) / 2
    radius_p = (tesseroid[4] + tesseroid[5]) / 2
    # Convert angles to radians
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance = np.sqrt(radius ** 2 + radius_p ** 2 - 2 * radius * radius_p * cospsi)
    return distance
