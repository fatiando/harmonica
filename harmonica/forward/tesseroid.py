"""
Forward modelling for tesseroids
"""
import numpy as np
from numba import jit
from numpy.polynomial.legendre import leggauss

from ..constants import GRAVITATIONAL_CONST
from .point_mass import jit_point_masses_gravity, kernel_potential, kernel_g_radial

STACK_SIZE = 100
MAX_DISCRETIZATIONS = 100000
GLQ_DEGREES = [2, 2, 2]
DISTANCE_SIZE_RATII = {"potential": 1, "g_radial": 2.5}
KERNELS = {"potential": kernel_potential, "g_radial": kernel_g_radial}


def tesseroid_gravity(
    coordinates,
    tesseroid,
    density,
    field,
    distance_size_ratii=DISTANCE_SIZE_RATII,
    glq_degrees=GLQ_DEGREES,
    stack_size=STACK_SIZE,
    max_discretizations=MAX_DISCRETIZATIONS,
    three_dimensional_adaptive_discretization=False,
):
    """
    Compute gravitational field of a tesseroid on a single computation point

    Parameters
    ----------
    coordinates: list or 1d-array
        List or array containing `longitude`, `latitude` and `radius` of a single
        computation point defined on a spherical geocentric coordinate system.
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
        - Radial acceleration: ``g_radial``
    """
    if field not in KERNELS:
        raise ValueError("Gravity field {} not recognized".format(field))
    # Get value of D (distance_size_ratio)
    distance_size_ratio = distance_size_ratii[field]
    # Convert coordinates and tesseroid to array to make Numba run only on Numpy arrays
    tesseroid = np.array(tesseroid)
    coordinates = np.array(coordinates)
    # Sanity checks for tesseroid
    _check_tesseroid(tesseroid)
    # Initialize arrays to perform memory allocation only once
    stack = np.empty((stack_size, 6))
    small_tesseroids = np.empty((max_discretizations, 6))
    # Apply adaptive discretization on tesseroid
    n_splits, error = _adaptive_discretization(
        coordinates,
        tesseroid,
        distance_size_ratio,
        stack,
        small_tesseroids,
        radial_discretization=three_dimensional_adaptive_discretization,
    )
    if error == -1:
        raise OverflowError("Stack Overflow. Try to increase the stack size.")
    elif error == -2:
        raise OverflowError(
            "Small Tesseroids Overflow. Try to increase the maximum number of splits."
        )
    # Get GLQ unscaled nodes, weights and number of point masses per small tesseroid
    n_point_masses, glq_nodes, glq_weights = glq_nodes_weights(glq_degrees)
    # Get total number of point masses and initialize arrays
    n_point_masses *= n_splits
    point_masses = np.empty((3, n_point_masses))
    weights = np.empty(n_point_masses)
    # Get equivalent point masses
    tesseroids_to_point_masses(
        small_tesseroids[:n_splits], glq_nodes, glq_weights, point_masses, weights
    )
    # Compute gravity fields
    longitude_p, latitude_p, radius_p = (i.ravel() for i in point_masses[:3])
    masses = density * weights
    result = jit_point_masses_gravity(
        coordinates, longitude_p, latitude_p, radius_p, masses, KERNELS[field]
    )
    result *= GRAVITATIONAL_CONST
    # Convert to more convenient units
    if field == "g_radial":
        result *= 1e5  # SI to mGal
    return result


@jit(nopython=True)
def tesseroids_to_point_masses(
    tesseroids, glq_nodes, glq_weights, point_masses, weights
):
    """
    Convert tesseroids to equivalent point masses on nodes of GLQ
    """
    # Unpack nodes and weights
    lon_nodes, lat_nodes, rad_nodes = glq_nodes[:]
    lon_weights, lat_weights, rad_weights = glq_weights[:]
    # Recover GLQ degrees from nodes
    lon_glq_degree = len(lon_nodes)
    lat_glq_degree = len(lat_nodes)
    rad_glq_degree = len(rad_nodes)
    # Convert each tesseroid to a point mass
    mass_index = 0
    for i in range(len(tesseroids)):
        w = tesseroids[i, 0]
        e = tesseroids[i, 1]
        s = tesseroids[i, 2]
        n = tesseroids[i, 3]
        bottom = tesseroids[i, 4]
        top = tesseroids[i, 5]
        A_factor = 1 / 8 * np.radians(e - w) * np.radians(n - s) * (top - bottom)
        for i in range(lon_glq_degree):
            for j in range(lat_glq_degree):
                for k in range(rad_glq_degree):
                    # Compute coordinates of each point mass
                    longitude = 0.5 * (e - w) * lon_nodes[i] + 0.5 * (e + w)
                    latitude = 0.5 * (n - s) * lat_nodes[j] + 0.5 * (n + s)
                    radius = 0.5 * (top - bottom) * rad_nodes[k] + 0.5 * (top + bottom)
                    kappa = radius ** 2 * np.cos(np.radians(latitude))
                    point_masses[0, mass_index] = longitude
                    point_masses[1, mass_index] = latitude
                    point_masses[2, mass_index] = radius
                    weights[mass_index] = (
                        A_factor
                        * kappa
                        * lon_weights[i]
                        * lat_weights[j]
                        * rad_weights[k]
                    )
                    mass_index += 1


def glq_nodes_weights(glq_degrees):
    """
    Calculate 3D GLQ unscaled nodes, weights and number of point masses
    """
    # Unpack GLQ degrees
    lon_degree, lat_degree, rad_degree = glq_degrees[:]
    # Get number of point masses
    n_point_masses = np.prod(glq_degrees)
    # Get nodes coordinates and weights
    lon_node, lon_weights = leggauss(lon_degree)
    lat_node, lat_weights = leggauss(lat_degree)
    rad_node, rad_weights = leggauss(rad_degree)
    # Reorder nodes and weights
    glq_nodes = [lon_node, lat_node, rad_node]
    glq_weights = [lon_weights, lat_weights, rad_weights]
    return n_point_masses, glq_nodes, glq_weights


@jit(nopython=True)
def _adaptive_discretization(
    coordinates,
    tesseroid,
    distance_size_ratio,
    stack,
    small_tesseroids,
    radial_discretization=False,
):
    """
    Three or two dimensional adaptive discretization
    """
    # Create stack of tesseroids
    stack[0] = tesseroid
    stack_top = 0
    error = 0
    n_splits = 0
    while stack_top >= 0:
        # Pop the first tesseroid from the stack
        tesseroid = stack[stack_top]
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
            # Raise error if stack overflow
            if stack_top + n_lon * n_lat * n_rad > stack.shape[0]:
                error = -1
                return n_splits, error
            stack_top = _split_tesseroid(
                tesseroid, n_lon, n_lat, n_rad, stack, stack_top
            )
        else:
            # Raise error if small_tesseroids overflow
            if n_splits + 1 > small_tesseroids.shape[0]:
                error = -2
                return n_splits, error
            small_tesseroids[n_splits] = tesseroid
            n_splits += 1
    return n_splits, error


@jit(nopython=True)
def _split_tesseroid(tesseroid, n_lon, n_lat, n_rad, stack, stack_top):
    """
    Split tesseroid along each dimension
    """
    w, e, s, n, bottom, top = tesseroid[:]
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
    # Get coordinates of computation point
    longitude, latitude, radius = coordinates[:]
    # Get center of the tesseroid
    w, e, s, n, bottom, top = tesseroid[:]
    longitude_p = (w + e) / 2
    latitude_p = (s + n) / 2
    radius_p = (bottom + top) / 2
    # Convert angles to radians
    longitude, latitude = np.radians(longitude), np.radians(latitude)
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    # Compute distance
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance = np.sqrt(radius ** 2 + radius_p ** 2 - 2 * radius * radius_p * cospsi)
    return distance


@jit(nopython=True)
def _check_tesseroid(tesseroid):
    "Check if tesseroid boundaries are well defined"
    w, e, s, n, bottom, top = tesseroid[:]
    if w >= e:
        raise ValueError(
            "Invalid tesseroid: {} (W, E, S, N, BOTTOM, TOP). ".format(tesseroid)
            + "W must be lower than E."
        )
    if s >= n:
        raise ValueError(
            "Invalid tesseroid: {} (W, E, S, N, BOTTOM, TOP). ".format(tesseroid)
            + "S must be lower than N."
        )
    if bottom > top:
        raise ValueError(
            "Invalid tesseroid: {} (W, E, S, N, BOTTOM, TOP). ".format(tesseroid)
            + "BOTTOM must be lower than TOP."
        )
    if bottom < 0 or top < 0:
        raise ValueError(
            "Invalid tesseroid: {} (W, E, S, N, BOTTOM, TOP). ".format(tesseroid)
            + "BOTTOM and TOP radii must be greater than zero."
        )
