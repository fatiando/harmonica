# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utils functions for tesseroids with variable density
"""
import numpy as np
from numba import jit
from scipy.optimize import minimize_scalar

DELTA_RATIO = 0.1


@jit(nopython=True)
def gauss_legendre_quadrature_variable_density(
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
    density : func
        Density func of the tesseroid in SI units.
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
            density_p = density(radius_p)
            # Get kappa constant for the point mass
            kappa = radius_p ** 2 * cosphi_p
            for i, lon_node in enumerate(lon_nodes):
                # Get the longitude of the point mass
                longitude_p = np.radians(0.5 * (e - w) * lon_node + 0.5 * (e + w))
                # Compute the mass of the point mass
                mass = (
                    density_p
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


# Density-based discretization functions
# --------------------------------------
def density_based_discretization(tesseroids, density):
    """
    Apply density_based discretization to a collection of tesseroids

    Parameters
    ----------
    tesseroids : 2d-array
        Array containing the coordinates of the tesseroid. Each row of the
        array should contain the boundaries of each tesseroid in the following
        order: ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        The longitudinal and latitudinal boundaries should be in degrees, while
        the radial ones must be in meters.
    density : func
        Continuous density function of the tesseroid in SI units.

    Returns
    -------
    discretized_tesseroids : 2d-array
        Array containing the coordinates of radially discretized tesseriods.
        Each row of the array will have the boundaries for each new tesseroid.
    """
    discretized_tesseroids = []
    for tesseroid in tesseroids:
        discretized_tesseroids.extend(_density_based_discretization(tesseroid, density))
    return np.atleast_2d(discretized_tesseroids)


def _density_based_discretization(tesseroid, density):
    """
    Applies density-based discretization to a single tesseroid

    Splits the tesseroid on the points of maximum density variance

    Parameters
    ----------
    tesseroid : tuple
        Tuple containing the boundaries of the tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        Horizontal boundaries should be in degrees and radial boundaries in
        meters.
    density : func
        Density func of the tesseroid in SI units.

    Returns
    -------
    tesseroids : list
        List containing the boundaries of discretized tesseroids.
    """
    # Define normalized density
    def normalized_density(radius):
        return (density(radius) - density_min) / (density_max - density_min)

    # Get boundaries of original tesseroid
    w, e, s, n, bottom, top = tesseroid[:]
    # Get minimum and maximum values of the density
    density_min, density_max = density_minmax(density, bottom, top)
    # Return the original tesseroid if max and min densities are equal
    if np.isclose(density_min, density_max):
        return [tesseroid]
    # Store the size of the original tesseroid
    size_original_tesseroid = top - bottom
    # Initialize list of pending and output tesseroids
    pending, tesseroids = [tesseroid], []
    # Discretization of the tesseroid
    while pending:
        tesseroid = pending.pop(0)
        bottom, top = tesseroid[-2:]
        radius_split, max_diff = maximum_absolute_diff(normalized_density, bottom, top)
        size_ratio = (top - bottom) / size_original_tesseroid
        if max_diff * size_ratio > DELTA_RATIO:
            pending.append([w, e, s, n, radius_split, top])
            pending.append([w, e, s, n, bottom, radius_split])
        else:
            tesseroids.append([w, e, s, n, bottom, top])
    return tesseroids


def density_minmax(density, bottom, top):
    """
    Compute the minimum and maximum value of a bounded density
    """
    # Calculate min and max density values at the top and bottom boundaries
    density_bounds_min, density_bounds_max = np.sort([density(bottom), density(top)])
    # Estimate the minimum value of the density function withing bounds
    kwargs = dict(bounds=[bottom, top], method="bounded")
    minimum = np.min(
        (
            minimize_scalar(density, **kwargs).fun,
            density_bounds_min,
        )
    )
    # Estimate the maximum value of the density function withing bounds
    maximum = np.max(
        (
            -minimize_scalar(lambda radius: -density(radius), **kwargs).fun,
            density_bounds_max,
        )
    )
    return minimum, maximum


def maximum_absolute_diff(normalized_density, bottom, top):
    """
    Compute maximum abs difference between normalized density and straight line

    The maximum difference is computed within the ``bottom`` and ``top``
    boundaries.
    """

    def neg_absolute_difference(radius):
        """
        Define minus absolute diff between normalized density and straight line
        """
        return -np.abs(
            normalized_density(radius)
            - straight_line(radius, normalized_density, bottom, top)
        )

    # Use scipy.optimize.minimize_scalar for maximizing the absolute difference
    result = minimize_scalar(
        neg_absolute_difference,
        bounds=[bottom, top],
        method="bounded",
    )
    # Get maximum difference and the radius at which it takes place
    radius_split = result.x
    max_diff = -result.fun
    return radius_split, max_diff


def straight_line(radius, normalized_density, bottom, top):
    """
    Compute the reference straight line that joins points of normalized density
    """
    norm_density_bottom = normalized_density(bottom)
    norm_density_top = normalized_density(top)
    slope = (norm_density_top - norm_density_bottom) / (top - bottom)
    return slope * (radius - bottom) + norm_density_bottom
