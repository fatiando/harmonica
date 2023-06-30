# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for tesseroids
"""
import numpy as np
from numba import jit, prange

from ..constants import GRAVITATIONAL_CONST
from ._tesseroid_utils import (
    _adaptive_discretization,
    _check_points_outside_tesseroids,
    _check_tesseroids,
    _discard_null_tesseroids,
    gauss_legendre_quadrature,
    glq_nodes_weights,
)
from ._tesseroid_variable_density import (
    density_based_discretization,
    gauss_legendre_quadrature_variable_density,
)
from .point import kernel_g_z_spherical, kernel_potential_spherical

STACK_SIZE = 100
MAX_DISCRETIZATIONS = 100000
GLQ_DEGREES = (2, 2, 2)
DISTANCE_SIZE_RATII = {"potential": 1, "g_z": 2.5}


def tesseroid_gravity(
    coordinates,
    tesseroids,
    density,
    field,
    parallel=True,
    radial_adaptive_discretization=False,
    dtype=np.float64,
    disable_checks=False,
):
    r"""
    Compute gravitational field of tesseroids on computation points.

    .. warning::

        The ``g_z`` field returns the downward component of the gravitational
        acceleration on the local North oriented coordinate system.
        It is equivalent to the opposite of the radial component, therefore
        it's positive if the acceleration vector points inside the spheroid.

    .. important::

        - The gravitational potential is returned in :math:`\text{J}/\text{kg}`.
        - The gravity acceleration components are returned in mgal
          (:math:`\text{m}/\text{s}^2`).

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``longitude``, ``latitude`` and
        ``radius`` coordinates of the computation points, defined on
        a spherical geocentric coordinate system. Both ``longitude`` and
        ``latitude`` should be in degrees and ``radius`` in meters.
    tesseroids : list or 2d-array
        List or array containing the coordinates of the tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The longitudinal and latitudinal boundaries should be in degrees, while
        the radial ones must be in meters.
    density : list, array or func decorated with :func:`numba.jit`
        List or array containing the density of each tesseroid in kg/m^3.
        Alternatively, it can be a continuous function within the boundaries of
        the tesseroids, in which case the variable density tesseroids method
        introduced in [Soler2019]_ will be used.
        If ``density`` is a function, it should be decorated with
        :func:`numba.jit`.
    field : str
        Gravitational field that wants to be computed.
        The available fields are:

        - Gravitational potential: ``potential``
        - Downward acceleration: ``g_z``

    parallel : bool (optional)
        If True the computations will run in parallel using Numba built-in
        parallelization. If False, the forward model will run on a single core.
        Might be useful to disable parallelization if the forward model is run
        by an already parallelized workflow. Default to True.
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
        points. Gravitational potential is returned in
        :math:`\text{J}/\text{kg}` and acceleration components in mgal.

    References
    ----------
    [Soler2019]_

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
    >>> g_z = tesseroid_gravity(coordinates, tesseroid, density, field="g_z")

    >>> # Define a linear density function for the same tesseroid.
    >>> # It should be decorated with numba.njit
    >>> from numba import jit
    >>> @jit
    ... def linear_density(radius):
    ...     density_top = 2670.
    ...     density_bottom = 3300.
    ...     slope = (density_top - density_bottom) / (top - bottom)
    ...     return slope * (radius - bottom) + density_bottom
    >>> # Compute the downward acceleration it generates
    >>> g_z = tesseroid_gravity(
    ...     coordinates, tesseroid, linear_density, field="g_z"
    ... )

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
    # Sanity checks for tesseroids and computation points
    if not disable_checks:
        tesseroids = _check_tesseroids(tesseroids)
        _check_points_outside_tesseroids(coordinates, tesseroids)
    # Check if density is a function or constant values
    if callable(density):
        # Run density-based discretization on each tesseroid
        tesseroids = density_based_discretization(tesseroids, density)
    else:
        density = np.atleast_1d(density).ravel()
        if not disable_checks and density.size != tesseroids.shape[0]:
            raise ValueError(
                "Number of elements in density ({}) ".format(density.size)
                + "mismatch the number of tesseroids ({})".format(tesseroids.shape[0])
            )
        # Discard null tesseroids (zero density or zero volume)
        tesseroids, density = _discard_null_tesseroids(tesseroids, density)
    # Get GLQ unscaled nodes, weights and number of nodes for each small
    # tesseroid
    glq_nodes, glq_weights = glq_nodes_weights(GLQ_DEGREES)
    # Compute gravitational field
    dispatcher(parallel, density)(
        coordinates,
        tesseroids,
        density,
        result,
        DISTANCE_SIZE_RATII[field],
        radial_adaptive_discretization,
        glq_nodes,
        glq_weights,
        kernels[field],
        dtype,
    )
    result *= GRAVITATIONAL_CONST
    # Convert to more convenient units
    if field == "g_z":
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


def dispatcher(parallel, density):
    """
    Return the jitted compiled forward modelling function

    The choice of the forward modelling function is based on whether the
    density is a function and if the model should be run in parallel.
    """
    if callable(density):
        dispatchers = {
            True: jit_tesseroid_gravity_variable_density_parallel,
            False: jit_tesseroid_gravity_variable_density_serial,
        }
    else:
        dispatchers = {
            True: jit_tesseroid_gravity_parallel,
            False: jit_tesseroid_gravity_serial,
        }
    return dispatchers[parallel]


def jit_tesseroid_gravity(
    coordinates,
    tesseroids,
    density,
    result,
    distance_size_ratio,
    radial_adaptive_discretization,
    glq_nodes,
    glq_weights,
    kernel,
    dtype,
):
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
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius``
        in meters.
    tesseroids : 2d-array
        Array containing the boundaries of each tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.
        All tesseroids must have valid boundary coordinates.
        Horizontal boundaries should be in degrees while radial boundaries
        should be in meters.
    density : 1d-array
        Density of each tesseroid in SI units.
    result : 1d-array
        Array where the gravitational effect of each tesseroid will be added.
    distance_size_ratio : float
        Value of the distance size ratio.
    radial_adaptive_discretization : bool
        If ``False``, the adaptive discretization algorithm will split the
        tesseroid only on the horizontal direction.
        If ``True``, it will perform a three dimensional adaptive
        discretization, splitting the tesseroids on every direction.
    glq_nodes : list
        List containing unscaled GLQ nodes.
    glq_weights : list
        List containing GLQ weights of the nodes.
    kernel : func
        Kernel function for the gravitational field of point masses.
    dtype : data-type
        Data type assigned to the resulting gravitational field.
    """
    # Get coordinates of the observation points
    # and precompute trigonometric functions
    longitude, latitude, radius = coordinates[:]
    longitude_rad = np.radians(longitude)
    cosphi = np.cos(np.radians(latitude))
    sinphi = np.sin(np.radians(latitude))
    # Loop over computation points
    for l in prange(longitude.size):
        # Initialize arrays to perform memory allocation only once
        stack = np.empty((STACK_SIZE, 6), dtype=dtype)
        small_tesseroids = np.empty((MAX_DISCRETIZATIONS, 6), dtype=dtype)
        # Loop over tesseroids
        for m in range(tesseroids.shape[0]):
            # Apply adaptive discretization on tesseroid
            n_splits = _adaptive_discretization(
                (longitude[l], latitude[l], radius[l]),
                tesseroids[m, :],
                distance_size_ratio,
                stack,
                small_tesseroids,
                radial_adaptive_discretization,
            )
            # Compute effect of the tesseroid through GLQ
            for tess_index in range(n_splits):
                tesseroid = small_tesseroids[tess_index, :]
                result[l] += gauss_legendre_quadrature(
                    longitude_rad[l],
                    cosphi[l],
                    sinphi[l],
                    radius[l],
                    tesseroid,
                    density[m],
                    glq_nodes,
                    glq_weights,
                    kernel,
                )


def jit_tesseroid_gravity_variable_density(
    coordinates,
    tesseroids,
    density,
    result,
    distance_size_ratio,
    radial_adaptive_discretization,
    glq_nodes,
    glq_weights,
    kernel,
    dtype,
):
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
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius``
        in meters.
    tesseroids : 2d-array
        Array containing the boundaries of each tesseroid:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top`` under a geocentric
        spherical coordinate system.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.
        All tesseroids must have valid boundary coordinates.
        Horizontal boundaries should be in degrees while radial boundaries
        should be in meters.
    density : func
        Density function of every tesseroid in SI units.
    result : 1d-array
        Array where the gravitational effect of each tesseroid will be added.
    distance_size_ratio : float
        Value of the distance size ratio.
    radial_adaptive_discretization : bool
        If ``False``, the adaptive discretization algorithm will split the
        tesseroid only on the horizontal direction.
        If ``True``, it will perform a three dimensional adaptive
        discretization, splitting the tesseroids on every direction.
    glq_nodes : list
        List containing unscaled GLQ nodes.
    glq_weights : list
        List containing GLQ weights of the nodes.
    kernel : func
        Kernel function for the gravitational field of point masses.
    dtype : data-type
        Data type assigned to the resulting gravitational field.
    """
    # Get coordinates of the observation points
    # and precompute trigonometric functions
    longitude, latitude, radius = coordinates[:]
    longitude_rad = np.radians(longitude)
    cosphi = np.cos(np.radians(latitude))
    sinphi = np.sin(np.radians(latitude))
    # Loop over computation points
    for l in prange(longitude.size):
        # Initialize arrays to perform memory allocation only once
        stack = np.empty((STACK_SIZE, 6), dtype=dtype)
        small_tesseroids = np.empty((MAX_DISCRETIZATIONS, 6), dtype=dtype)
        # Loop over tesseroids
        for m in range(tesseroids.shape[0]):
            # Apply adaptive discretization on tesseroid
            n_splits = _adaptive_discretization(
                (longitude[l], latitude[l], radius[l]),
                tesseroids[m, :],
                distance_size_ratio,
                stack,
                small_tesseroids,
                radial_adaptive_discretization,
            )
            # Compute effect of the tesseroid through GLQ
            for tess_index in range(n_splits):
                tesseroid = small_tesseroids[tess_index, :]
                result[l] += gauss_legendre_quadrature_variable_density(
                    longitude_rad[l],
                    cosphi[l],
                    sinphi[l],
                    radius[l],
                    tesseroid,
                    density,
                    glq_nodes,
                    glq_weights,
                    kernel,
                )


# Define jitted versions of the forward modelling function
jit_tesseroid_gravity_serial = jit(nopython=True)(jit_tesseroid_gravity)
jit_tesseroid_gravity_parallel = jit(nopython=True, parallel=True)(
    jit_tesseroid_gravity
)
jit_tesseroid_gravity_variable_density_serial = jit(nopython=True)(
    jit_tesseroid_gravity_variable_density
)
jit_tesseroid_gravity_variable_density_parallel = jit(nopython=True, parallel=True)(
    jit_tesseroid_gravity_variable_density
)
