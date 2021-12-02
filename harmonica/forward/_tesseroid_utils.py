# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utils functions for tesseroid forward modelling
"""
import numpy as np
from numba import jit
from numpy.polynomial.legendre import leggauss

from .utils import distance_spherical


@jit(nopython=True)
def gauss_legendre_quadrature(
    longitude,
    cosphi,
    sinphi,
    radius,
    tesseroid,
    density,
    glq_nodes,
    glq_weights,
    kernel,
):
    r"""
    Compute the effect of a tesseroid on a single observation point through GLQ

    The tesseroid is converted into a set of point masses located on the
    scaled nodes of the Gauss-Legendre Quadrature. The number of point masses
    created from each tesseroid is equal to the product of the GLQ degrees for
    each direction (:math:`N_r`, :math:`N_\lambda`, :math:`N_\phi`). The mass
    of each point mass is defined as the product of the tesseroid density
    (:math:`\rho`), the GLQ weights for each direction (:math:`W_i^r`,
    :math:`W_j^\phi`, :math:`W_k^\lambda`), the scale constant :math:`A` and
    the :math:`\kappa` factor evaluated on the coordinates of the point mass.

    Parameters
    ----------
    longitude : float
        Longitudinal coordinate of the observation points in radians.
    cosphi : float
        Cosine of the latitudinal coordinate of the observation point in
        radians.
    sinphi : float
        Sine of the latitudinal coordinate of the observation point in
        radians.
    radius : float
        Radial coordinate of the observation point in meters.
    tesseroids : 1d-array
        Array containing the boundaries of the tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        Horizontal boundaries should be in degrees and radial boundaries in
        meters.
    density : float
        Density of the tesseroid in SI units.
    glq_nodes : list
        Unscaled location of GLQ nodes for each direction.
    glq_weights : list
        GLQ weigths for each node for each direction.
    kernel : func
        Kernel function for the gravitational field of point masses.

    """
    # Get tesseroid boundaries
    w, e, s, n, bottom, top = tesseroid[:]
    # Calculate the A factor for the tesseroid
    a_factor = 1 / 8 * np.radians(e - w) * np.radians(n - s) * (top - bottom)
    # Unpack nodes and weights
    lon_nodes, lat_nodes, rad_nodes = glq_nodes[:]
    lon_weights, lat_weights, rad_weights = glq_weights[:]
    # Compute effect of the tesseroid on the observation point
    # by iterating over the location of the point masses
    # (move the iteration along the longitudinal nodes to the bottom for
    # optimization: reduce the number of times that the trigonometric functions
    # are evaluated)
    result = 0.0
    for j, lat_node in enumerate(lat_nodes):
        # Get the latitude of the point mass
        latitude_p = np.radians(0.5 * (n - s) * lat_node + 0.5 * (n + s))
        cosphi_p = np.cos(latitude_p)
        sinphi_p = np.sin(latitude_p)
        for k, rad_node in enumerate(rad_nodes):
            # Get the radius of the point mass
            radius_p = 0.5 * (top - bottom) * rad_node + 0.5 * (top + bottom)
            # Get kappa constant for the point mass
            kappa = radius_p ** 2 * cosphi_p
            for i, lon_node in enumerate(lon_nodes):
                # Get the longitude of the point mass
                longitude_p = np.radians(0.5 * (e - w) * lon_node + 0.5 * (e + w))
                # Compute the mass of the point mass
                mass = (
                    density
                    * a_factor
                    * kappa
                    * lon_weights[i]
                    * lat_weights[j]
                    * rad_weights[k]
                )
                # Add effect of the current point mass to the result
                result += mass * kernel(
                    longitude,
                    cosphi,
                    sinphi,
                    radius,
                    longitude_p,
                    cosphi_p,
                    sinphi_p,
                    radius_p,
                )
    return result


def glq_nodes_weights(glq_degrees):
    """
    Calculate GLQ unscaled nodes and weights

    Parameters
    ----------
    glq_degrees : list
        List of GLQ degrees for each direction: ``longitude``, ``latitude``,
        ``radius``.

    Returns
    -------
    glq_nodes : list
        Unscaled GLQ nodes for each direction: ``longitude``, ``latitude``,
        ``radius``.
    glq_weights : list
        GLQ weights for each node on each direction: ``longitude``,
        ``latitude``, ``radius``.
    """
    # Unpack GLQ degrees
    lon_degree, lat_degree, rad_degree = glq_degrees[:]
    # Get nodes coordinates and weights
    lon_node, lon_weights = leggauss(lon_degree)
    lat_node, lat_weights = leggauss(lat_degree)
    rad_node, rad_weights = leggauss(rad_degree)
    # Reorder nodes and weights
    glq_nodes = (lon_node, lat_node, rad_node)
    glq_weights = (lon_weights, lat_weights, rad_weights)
    return glq_nodes, glq_weights


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
        Array with shape ``(6, MAX_DISCRETIZATIONS)`` that will contain every
        small tesseroid produced by the adaptive discretization algorithm.
        If too many discretizations will take place, increase the
        ``MAX_DISCRETIZATIONS``.
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
                    + " Please increase the MAX_DISCRETIZATIONS."
                )
            small_tesseroids[n_splits] = tesseroid
            n_splits += 1
    return n_splits


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
                stack_top += 1  # noqa: SIM113, don't want to use enumerate here
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


def _check_tesseroids(tesseroids):
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


def _check_points_outside_tesseroids(coordinates, tesseroids):
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
