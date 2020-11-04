# Copyright (c) YEAR The PROJECT Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for tesseroids
"""
from numba import jit
import numpy as np
from numpy.polynomial.legendre import leggauss

from ..constants import GRAVITATIONAL_CONST
from .utils import distance_spherical
from .point_mass import (
    jit_point_mass_spherical,
    kernel_potential_spherical,
    kernel_g_z_spherical,
)

STACK_SIZE = 100
MAX_DISCRETIZATIONS = 100000
GLQ_DEGREES = (2, 2, 2)
DISTANCE_SIZE_RATII = {"potential": 1, "g_z": 2.5}


def tesseroid_gravity(
    coordinates,
    tesseroids,
    density,
    field,
    distance_size_ratii=None,
    glq_degrees=GLQ_DEGREES,
    stack_size=STACK_SIZE,
    max_discretizations=MAX_DISCRETIZATIONS,
    radial_adaptive_discretization=False,
    dtype=np.float64,
    disable_checks=False,
):  # pylint: disable=too-many-locals, too-many-arguments
    """
    Compute gravitational field of tesseroids on computation points.

    .. warning::

        The ``g_z`` field returns the downward component of the gravitational
        acceleration on the local North oriented coordinate system.
        It is equivalent to the opposite of the radial component, therefore
        it's positive if the acceleration vector points inside the spheroid.

    Parameters
    ----------
    coordinates : list or 1d-array
        List or array containing ``longitude``, ``latitude`` and ``radius`` of
        the computation points defined on a spherical geocentric coordinate
        system.
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius``
        in meters.
    tesseroids : list or 1d-array
        List or array containing the coordinates of the tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The longitudinal and latitudinal boundaries should be in degrees, while
        the radial ones must be in meters.
    density : list or array
        List or array containing the density of each tesseroid in kg/m^3.
    field : str
        Gravitational field that wants to be computed.
        The available fields are:

        - Gravitational potential: ``potential``
        - Downward acceleration: ``g_z``

    distance_size_ratio : dict or None (optional)
        Dictionary containing distance-size ratii for each gravitational field
        used on the adaptive discretization algorithm.
        Values must be the available fields and keys should be the desired
        distance-size ratio.
        The greater the distance-size ratio, more discretizations will occur,
        increasing the accuracy of the numerical approximation but also the
        computation time.
        If None, the default values of distance-size ratii will be used:
        D = 1 for the potential and D = 2.5 for the gradient.
        Default to None.
    glq_degrees : tuple (optional)
        List containing the GLQ degrees used on each direction:
        ``glq_degree_longitude``, ``glq_degree_latitude``,
        ``glq_degree_radius``.
        The GLQ degree specifies how many point masses will be created along
        each direction.
        Increasing the GLQ degree will increase the accuracy of the numerical
        approximation, but also the computation time.
        Default ``[2, 2, 2]``.
    stack_size : int (optional)
        Size of the tesseroid stack used on the adaptive discretization
        algorithm.
        If the algorithm will perform too many splits, please increase the
        stack size.
    max_discretizations : int (optional)
        Maximum number of splits made by the adaptive discretization algorithm.
        If the algorithm will perform too many splits, please increase the
        maximum number of splits.
    radial_adaptive_discretization : bool (optional)
        If ``False``, the adaptive discretization algorithm will split the
        tesseroid only on the horizontal direction.
        If ``True``, it will perform a three dimensional adaptive
        discretization, splitting the tesseroids on every direction.
        Default ``False``.
    dtype : data-type (optional)
        Data type assigned to the resulting gravitational field. Default to
        ``np.float64``.
    disable_checks : bool (optional)
        Flag that controls whether to perform a sanity check on the model.
        Should be set to ``True`` only when it is certain that the input model
        is valid and it does not need to be checked.
        Default to ``False``.

    Returns
    -------
    result : array
        Gravitational field generated by the tesseroids on the computation
        points.

    Examples
    --------

    >>> # Get WGS84 ellipsoid from the Boule package
    >>> import boule
    >>> ellipsoid = boule.WGS84
    >>> # Define tesseroid of 1km of thickness with top surface on the mean
    >>> # Earth radius
    >>> thickness = 1000
    >>> top = ellipsoid.mean_radius
    >>> bottom = top - thickness
    >>> w, e, s, n = -1.0, 1.0, -1.0, 1.0
    >>> tesseroid = [w, e, s, n, bottom, top]
    >>> # Set a density of 2670 kg/m^3
    >>> density = 2670.0
    >>> # Define computation point located on the top surface of the tesseroid
    >>> coordinates = [0, 0, ellipsoid.mean_radius]
    >>> # Compute radial component of the gravitational gradient in mGal
    >>> tesseroid_gravity(coordinates, tesseroid, density, field="g_z")
    array(112.54539933)

    """
    kernels = {"potential": kernel_potential_spherical, "g_z": kernel_g_z_spherical}
    if field not in kernels:
        raise ValueError("Gravitational field {} not recognized".format(field))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates, tesseroids and density to arrays
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    tesseroids = np.atleast_2d(tesseroids)
    density = np.atleast_1d(density).ravel()
    # Sanity checks for tesseroids and computation points
    if not disable_checks:
        if density.size != tesseroids.shape[0]:
            raise ValueError(
                "Number of elements in density ({}) ".format(density.size)
                + "mismatch the number of tesseroids ({})".format(tesseroids.shape[0])
            )
        tesseroids = _check_tesseroids(tesseroids)
        _check_points_outside_tesseroids(coordinates, tesseroids)
    # Get value of D (distance_size_ratio)
    if distance_size_ratii is None:
        distance_size_ratii = DISTANCE_SIZE_RATII
    if field not in distance_size_ratii:
        raise ValueError(
            'Gravitational field "{}" not found on distance_size_ratii dictionary'.format(
                field
            )
        )
    distance_size_ratio = distance_size_ratii[field]
    # Get GLQ unscaled nodes, weights and number of nodes for each small
    # tesseroid
    n_nodes, glq_nodes, glq_weights = glq_nodes_weights(glq_degrees)
    # Initialize arrays to perform memory allocation only once
    stack = np.empty((stack_size, 6), dtype=dtype)
    small_tesseroids = np.empty((max_discretizations, 6), dtype=dtype)
    point_masses = np.empty((3, n_nodes * max_discretizations), dtype=dtype)
    weights = np.empty(n_nodes * max_discretizations, dtype=dtype)
    # Compute gravitational field
    jit_tesseroid_gravity(
        coordinates,
        tesseroids,
        density,
        stack,
        small_tesseroids,
        point_masses,
        weights,
        result,
        distance_size_ratio,
        radial_adaptive_discretization,
        n_nodes,
        glq_nodes,
        glq_weights,
        kernels[field],
    )
    result *= GRAVITATIONAL_CONST
    # Convert to more convenient units
    if field == "g_z":
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


@jit(nopython=True, parallel=True)
def jit_tesseroid_gravity(
    coordinates,
    tesseroids,
    density,
    stack,
    small_tesseroids,
    point_masses,
    weights,
    result,
    distance_size_ratio,
    radial_discretization,
    n_nodes,
    glq_nodes,
    glq_weights,
    kernel,
):  # pylint: disable=too-many-locals,too-many-arguments,invalid-name
    """
    Compute gravitational field of tesseroids on computations points

    Perform adaptive discretization, convert each small tesseroid to equivalent
    point masses through GLQ and use point masses kernel functions to compute
    the gravitational field.

    Parameters
    ----------
    coordinates : tuple
        Tuple containing the coordinates of the computation points in spherical
        geocentric coordinate system in the following order:
        ``longitude``, ``latitude``, ``radius``.
        Each element of the tuple must be a 1d array.
    tesseroids : 2d-array
        Array containing the boundaries of each tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.
        All tesseroids must have valid boundary coordinates.
    density : 1d-array
        Density of each tesseroid in SI units.
    stack : 2d-array
        Empty array where tesseroids created by adaptive discretization
        algorithm will be processed.
    small_tesseroids : 2d-array
        Empty array where smaller tesseroids created by adaptive discretization
        algorithm will be stored.
    point_masses : 2d-array
        Empty array where equivalent point masses will be stored.
    weights : 1d-array
        Empty array where the GLQ weight of each point mass will be stored.
    result : 1d-array
        Array where the gravitational effect of each tesseroid will be added.
    distance_size_ratio : float
        Value of the distance size ratio.
    radial_discretization : bool
        If ``False``, the adaptive discretization algorithm will split the
        tesseroid only on the horizontal direction.
        If ``True``, it will perform a three dimensional adaptive
        discretization, splitting the tesseroids on every direction.
    n_nodes : int
        Total number of equivalent point masses that will be generated for each
        split tesseroid.
    glq_nodes : list
        List containing unscaled GLQ nodes.
    glq_weights : list
        List containing GLQ weights of the nodes.
    kernel : func
        Kernel function for the gravitational field of point masses.
    """
    for l in range(tesseroids.shape[0]):
        tesseroid = tesseroids[l, :]
        for m in range(coordinates[0].size):
            # Apply adaptive discretization on tesseroid
            n_splits = _adaptive_discretization(
                (coordinates[0][m], coordinates[1][m], coordinates[2][m]),
                tesseroid,
                distance_size_ratio,
                stack,
                small_tesseroids,
                radial_discretization,
            )
            # Get total number of point masses, their coordinates and weights
            n_point_masses = n_nodes * n_splits
            tesseroids_to_point_masses(
                small_tesseroids[:n_splits],
                glq_nodes,
                glq_weights,
                point_masses,
                weights,
            )
            # Compute gravitational fields
            jit_point_mass_spherical(
                coordinates[0][m : m + 1],  # slice lon to pass a single element array
                coordinates[1][m : m + 1],  # slice lat to pass a single element array
                coordinates[2][m : m + 1],  # slice rad to pass a single element array
                point_masses[0, :n_point_masses],
                point_masses[1, :n_point_masses],
                point_masses[2, :n_point_masses],
                density[l] * weights[:n_point_masses],
                result[m : m + 1],
                kernel,
            )


@jit(nopython=True)
def tesseroids_to_point_masses(
    tesseroids, glq_nodes, glq_weights, point_masses, weights
):  # pylint: disable=too-many-locals,invalid-name
    r"""
    Convert tesseroids to equivalent point masses on nodes of GLQ

    Each tesseroid is converted into a set of point masses located on the
    scaled nodes of the Gauss-Legendre Quadrature. The number of point masses
    created from each tesseroid is equal to the product of the GLQ degrees for
    each direction (:math:`N_r`, :math:`N_\lambda`, :math:`N_\phi`). It also
    compute a weight value for each point mass defined as the product of the
    GLQ weights for each direction (:math:`W_i^r`, :math:`W_j^\phi`,
    :math:`W_k^\lambda`), the scale constant :math:`A` and the :math:`\kappa`
    factor evaluated on the coordinates of the point mass.

    Parameters
    ----------
    tesseroids : 2d-array
        Array containing the boundaries of each tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.
        All tesseroids must have valid boundary coordinates.
    glq_nodes : list
        Unscaled location of GLQ nodes for each direction.
    glq_weights : list
        GLQ weigths for each node for each direction.
    point_masses : 2d-array
        Empty array with shape ``(3, n)``, where ``n`` is the total number of
        point masses computed as the product of number of tesseroids and the
        GLQ degrees for each direction.
        The location of the point masses will be located inside this array.
    weights : 1d-array
        Empty array with ``n`` elements.
        It will contain the weight constant for each point mass.

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
    for tesseroid_index in range(len(tesseroids)):
        w = tesseroids[tesseroid_index, 0]
        e = tesseroids[tesseroid_index, 1]
        s = tesseroids[tesseroid_index, 2]
        n = tesseroids[tesseroid_index, 3]
        bottom = tesseroids[tesseroid_index, 4]
        top = tesseroids[tesseroid_index, 5]
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
    Calculate GLQ unscaled nodes, weights and total number of nodes

    Parameters
    ----------
    glq_degrees : list
        List of GLQ degrees for each direction: ``longitude``, ``latitude``,
        ``radius``.

    Returns
    -------
    n_nodes : int
        Total number of nodes computed as the product of the GLQ degrees.
    glq_nodes : list
        Unscaled GLQ nodes for each direction: ``longitude``, ``latitude``,
        ``radius``.
    glq_weights : list
        GLQ weights for each node on each direction: ``longitude``,
        ``latitude``, ``radius``.
    """
    # Unpack GLQ degrees
    lon_degree, lat_degree, rad_degree = glq_degrees[:]
    # Get number of point masses
    n_nodes = np.prod(glq_degrees)
    # Get nodes coordinates and weights
    lon_node, lon_weights = leggauss(lon_degree)
    lat_node, lat_weights = leggauss(lat_degree)
    rad_node, rad_weights = leggauss(rad_degree)
    # Reorder nodes and weights
    glq_nodes = (lon_node, lat_node, rad_node)
    glq_weights = (lon_weights, lat_weights, rad_weights)
    return n_nodes, glq_nodes, glq_weights


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
    Perform the adaptive discretization algorithm on a tesseroid

    It apply the three or two dimensional adaptive discretization algorithm on
    a tesseroid after a single computation point.

    Parameters
    ----------
    coordinates : array
        Array containing ``longitude``, ``latitude`` and ``radius`` of a single
        computation point.
    tesseroid : array
        Array containing the boundaries of the tesseroid.
    distance_size_ratio : float
        Value for the distance-size ratio. A greater value will perform more
        discretizations.
    stack : 2d-array
        Array with shape ``(6, stack_size)`` that will temporarly hold the
        small tesseroids that are not yet processed.
        If too many discretizations will take place, increase the
        ``stack_size``.
    small_tesseroids : 2d-array
        Array with shape ``(6, max_discretizations)`` that will contain every
        small tesseroid produced by the adaptive discretization algorithm.
        If too many discretizations will take place, increase the
        ``max_discretizations``.
    radial_discretization : bool (optional)
        If ``True`` the three dimensional adaptive discretization will be
        applied.
        If ``False`` the two dimensional adaptive discretization will be
        applied, i.e. the tesseroid will only be split on the ``longitude`` and
        ``latitude`` directions.
        Default ``False``.

    Returns
    -------
    n_splits : int
        Total number of small tesseroids generated by the algorithm.
    """
    # Create stack of tesseroids
    stack[0] = tesseroid
    stack_top = 0
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
            # Number of tesseroids in stack = stack_top + 1
            if (stack_top + 1) + n_lon * n_lat * n_rad > stack.shape[0]:
                raise OverflowError("Stack Overflow. Try to increase the stack size.")
            stack_top = _split_tesseroid(
                tesseroid, n_lon, n_lat, n_rad, stack, stack_top
            )
        else:
            # Raise error if small_tesseroids overflow
            if n_splits + 1 > small_tesseroids.shape[0]:
                raise OverflowError(
                    "Exceeded maximum discretizations."
                    + " Please increase the maximum_discretizations."
                )
            small_tesseroids[n_splits] = tesseroid
            n_splits += 1
    return n_splits


@jit(nopython=True)
def _split_tesseroid(
    tesseroid, n_lon, n_lat, n_rad, stack, stack_top
):  # pylint: disable=too-many-locals
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
def _distance_tesseroid_point(
    coordinates, tesseroid
):  # pylint: disable=too-many-locals
    """
    Distance between a computation point and the center of a tesseroid
    """
    # Get center of the tesseroid
    w, e, s, n, bottom, top = tesseroid[:]
    longitude_p = (w + e) / 2
    latitude_p = (s + n) / 2
    radius_p = (bottom + top) / 2
    # Get distance between computation point and tesseroid center
    distance = distance_spherical(coordinates, (longitude_p, latitude_p, radius_p))
    return distance


def _check_tesseroids(tesseroids):  # pylint: disable=too-many-branches
    """
    Check if tesseroids boundaries are well defined

    A valid tesseroid should have:
        - latitudinal boundaries within the [-90, 90] degrees interval,
        - north boundaries greater or equal than the south boundaries,
        - radial boundaries positive or zero,
        - top boundaries greater or equal than the bottom boundaries,
        - longitudinal boundaries within the [-180, 360] degrees interval,
        - longitudinal interval must not be greater than one turn around the
          globe.

    Some valid tesseroids have its west boundary greater than the east one,
    e.g. ``(350, 10, ...)``. On these cases the ``_longitude_continuity``
    function is applied in order to move the longitudinal coordinates to the
    [-180, 180) interval. Any valid tesseroid should have east boundaries
    greater than the west boundaries before or after applying longitude
    continuity.

    Parameters
    ----------
    tesseroids : 2d-array
        Array containing the boundaries of the tesseroids in the following
        order: ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        Longitudinal and latitudinal boundaries must be in degrees.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.

    Returns
    -------
    tesseroids :  2d-array
        Array containing the boundaries of the tesseroids.
        If no longitude continuity needs to be applied, the returned array is
        the same one as the orignal.
        Otherwise, it's copied and its longitudinal boundaries are modified.
    """
    west, east, south, north, bottom, top = tuple(tesseroids[:, i] for i in range(6))
    err_msg = "Invalid tesseroid or tesseroids. "
    # Check if latitudinal boundaries are inside the [-90, 90] interval
    invalid = np.logical_or(
        np.logical_or(south < -90, south > 90), np.logical_or(north < -90, north > 90)
    )
    if (invalid).any():
        err_msg += (
            "The latitudinal boundaries must be inside the [-90, 90] "
            + "degrees interval.\n"
        )
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    # Check if south boundary is not greater than the corresponding north
    # boundary
    invalid = south > north
    if (invalid).any():
        err_msg += "The south boundary can't be greater than the north one.\n"
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    # Check if radial boundaries are positive or zero
    invalid = np.logical_or(bottom < 0, top < 0)
    if (invalid).any():
        err_msg += "The bottom and top radii should be positive or zero.\n"
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    # Check if top boundary is not greater than the corresponding bottom
    # boundary
    invalid = bottom > top
    if (invalid).any():
        err_msg += "The bottom radius boundary can't be greater than the top one.\n"
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    # Check if longitudinal boundaries are inside the [-180, 360] interval
    invalid = np.logical_or(
        np.logical_or(west < -180, west > 360), np.logical_or(east < -180, east > 360)
    )
    if (invalid).any():
        err_msg += (
            "The longitudinal boundaries must be inside the [-180, 360] "
            + "degrees interval.\n"
        )
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    # Apply longitude continuity if w > e
    if (west > east).any():
        tesseroids = _longitude_continuity(tesseroids)
        west, east, south, north, bottom, top = tuple(
            tesseroids[:, i] for i in range(6)
        )
    # Check if west boundary is not greater than the corresponding east
    # boundary, even after applying the longitude continuity
    invalid = west > east
    if (invalid).any():
        err_msg += "The west boundary can't be greater than the east one.\n"
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    # Check if the longitudinal interval is not grater than one turn around the
    # globe
    invalid = east - west > 360
    if (invalid).any():
        err_msg += (
            "The difference between east and west boundaries cannot be greater than "
            + "one turn around the globe.\n"
        )
        for tess in tesseroids[invalid]:
            err_msg += "\tInvalid tesseroid: {}\n".format(tess)
        raise ValueError(err_msg)
    return tesseroids


def _check_points_outside_tesseroids(
    coordinates, tesseroids
):  # pylint: disable=too-many-locals
    """
    Check if computation points are not inside the tesseroids

    Parameters
    ----------
    coordinates : 2d-array
        Array containing the coordinates of the computation points in the
        following order: ``longitude``, ``latitude`` and ``radius``.
        Both ``longitude`` and ``latitude`` must be in degrees.
        The array must have the following shape: (3, ``n_points``), where
        ``n_points`` is the total number of computation points.
    tesseroids : 2d-array
        Array containing the boundaries of the tesseroids in the following
        order: ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        Longitudinal and latitudinal boundaries must be in degrees.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.
        This array of tesseroids must have longitude continuity and valid
        boundaries.
        Run ``_check_tesseroids`` before.
    """
    longitude, latitude, radius = coordinates[:]
    west, east, south, north, bottom, top = tuple(tesseroids[:, i] for i in range(6))
    # Longitudinal boundaries of the tesseroid must be compared with
    # longitudinal coordinates of computation points when moved to
    # [0, 360) and [-180, 180).
    longitude_360 = longitude % 360
    longitude_180 = ((longitude + 180) % 360) - 180
    inside_longitude = np.logical_or(
        np.logical_and(
            west < longitude_360[:, np.newaxis], longitude_360[:, np.newaxis] < east
        ),
        np.logical_and(
            west < longitude_180[:, np.newaxis], longitude_180[:, np.newaxis] < east
        ),
    )
    inside_latitude = np.logical_and(
        south < latitude[:, np.newaxis], latitude[:, np.newaxis] < north
    )
    inside_radius = np.logical_and(
        bottom < radius[:, np.newaxis], radius[:, np.newaxis] < top
    )
    # Build array of booleans.
    # The (i, j) element is True if the computation point i is inside the
    # tesseroid j.
    inside = inside_longitude * inside_latitude * inside_radius
    if inside.any():
        err_msg = (
            "Found computation point inside tesseroid. "
            + "Computation points must be outside of tesseroids.\n"
        )
        for point_i, tess_i in np.argwhere(inside):
            err_msg += "\tComputation point '{}' found inside tesseroid '{}'\n".format(
                coordinates[:, point_i], tesseroids[tess_i, :]
            )
        raise ValueError(err_msg)


def _longitude_continuity(tesseroids):
    """
    Modify longitudinal boundaries of tesseroids to ensure longitude continuity

    Longitudinal boundaries of the tesseroids are moved to the ``[-180, 180)``
    degrees interval in case the ``west`` boundary is numerically greater than
    the ``east`` one.

    Parameters
    ----------
    tesseroids : 2d-array
        Longitudinal and latitudinal boundaries must be in degrees.
        Array containing the boundaries of each tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.

    Returns
    -------
    tesseroids : 2d-array
        Modified boundaries of the tesseroids.
    """
    # Copy the tesseroids to avoid modifying the original tesseroids array
    tesseroids = tesseroids.copy()
    west, east = tesseroids[:, 0], tesseroids[:, 1]
    tess_to_be_changed = west > east
    east[tess_to_be_changed] = ((east[tess_to_be_changed] + 180) % 360) - 180
    west[tess_to_be_changed] = ((west[tess_to_be_changed] + 180) % 360) - 180
    return tesseroids
