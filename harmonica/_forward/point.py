# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for point masses
"""

import numpy as np
from choclo.point import (
    gravity_e,
    gravity_ee,
    gravity_en,
    gravity_eu,
    gravity_n,
    gravity_nn,
    gravity_nu,
    gravity_pot,
    gravity_u,
    gravity_uu,
)
from numba import jit, prange

from choclo.constants import GRAVITATIONAL_CONST
from .utils import check_coordinate_system, distance_spherical_core


def point_gravity(
    coordinates,
    points,
    masses,
    field,
    coordinate_system="cartesian",
    parallel=True,
    dtype="float64",
):
    r"""
    Compute gravitational fields of point masses.

    It can compute the gravitational fields of point masses on a set of
    computation points defined either in Cartesian or geocentric spherical
    coordinates.

    The gravitational potential field generated by a point mass with mass
    :math:`m` located at a point :math:`Q` on a computation point :math:`P` can
    be computed as:

    .. math::

        V(P) = \frac{G m}{l},

    where :math:`G` is the gravitational constant and :math:`l` is the
    Euclidean distance between :math:`P` and :math:`Q` [Blakely1995]_.

    In Cartesian coordinates, the points :math:`P` and :math:`Q` are given by
    :math:`x`, :math:`y` and :math:`z` coordinates, which can be translated
    into ``northing``, ``easting`` and ``upward``, respectively. If :math:`P`
    is located at :math:`(x, y, z)`, and :math:`Q` at :math:`(x_p, y_p, z_p)`,
    the distance :math:`l` can be computed as:

    .. math::

        l = \sqrt{ (x - x_p)^2 + (y - y_p)^2 + (z - z_p)^2 }.

    The gradient of the potential, also known as the gravitational acceleration
    vector :math:`\vec{g}`, is defined as:

    .. math::

        \vec{g} = \nabla V

    and has components :math:`g_{northing}(P)`, :math:`g_{easting}(P)` and
    :math:`g_{upward}(P)` given by

    .. math::

        g_{northing}(P) = - \frac{G m}{l^3} (x - x_p),

    .. math::

        g_{easting}(P) = - \frac{G m}{l^3} (y - y_p)

    and

    .. math::

        g_{upward}(P) = - \frac{G m}{l^3} (z - z_p).

    We define the downward component of the gravitational acceleration as the
    opposite of :math:`g_{upward}` (remember that :math:`z` points upwards):

    .. math::

        g_{z}(P) = \frac{G m}{l^3} (z - z_p).

    On a geocentric spherical coordinate system, the points :math:`P` and
    :math:`Q` are given by the ``longitude``, ``latitude`` and ``radius``
    coordinates, i.e. :math:`\lambda`, :math:`\varphi` and :math:`r`,
    respectively. On this coordinate system, the Euclidean distance between
    :math:`P(r, \varphi, \lambda)` and :math:`Q(r_p, \varphi_p, \lambda_p)` can
    be calculated  as follows [Grombein2013]_:

    .. math::

        l = \sqrt{ r^2 + r_p^2 - 2 r r_p \cos \Psi },

    where

    .. math::

        \cos \Psi = \sin \varphi \sin \varphi_p +
        \cos \varphi \cos \varphi_p \cos(\lambda - \lambda_p).

    The radial component of the acceleration vector on a local North-oriented
    system whose origin is located on the point :math:`P(r, \varphi, \lambda)`
    is given by [Grombein2013]_:

    .. math::

        g_r(P) =  - \frac{G m}{l^3} (r - r_p \cos \Psi).

    We define the downward component of the gravitational acceleration
    :math:`g_z` as the opposite of the radial component:

    .. math::

        g_z(P) = \frac{G m}{l^3} (r - r_p \cos \Psi).

    .. warning::

        When working in Cartesian coordinates, the **z direction points
        upwards**, i.e. positive and negative values of ``upward`` represent
        points above and below the surface, respectively. But remember that the
        ``g_z`` field returns the downward component of the gravitational
        acceleration.

    .. warning::

        When working in geocentric spherical coordinates, remember that the
        ``g_z`` field returns the downward component of the gravitational
        acceleration on the local North oriented coordinate system. It is
        equivalent to the opposite of the radial component, therefore it's
        positive if the acceleration vector points inside the spheroid.


    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the coordinates of computation points in the
        following order: ``easting``, ``northing`` and ``upward`` (if
        coordinates given in Cartesian coordinates), or ``longitude``,
        ``latitude`` and ``radius`` (if given on a spherical geocentric
        coordinate system).
        All ``easting``, ``northing`` and ``upward`` should be in meters.
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius``
        in meters.
    points : list or array
        List or array containing the coordinates of the point masses in the
        following order: ``easting``, ``northing`` and ``upward`` (if
        coordinates given in Cartesian coordinates), or ``longitude``,
        ``latitude`` and ``radius`` (if given on a spherical geocentric
        coordinate system).
        All ``easting``, ``northing`` and ``upward`` should be in meters.
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius``
        in meters.
    masses : list or array
        List or array containing the mass of each point mass in kg.
    field : str
        Gravitational field that wants to be computed.
        The available fields coordinates are:

        - Gravitational potential: ``potential``
        - Downward acceleration: ``g_z``
        - Northing acceleration: ``g_northing``
        - Easting acceleration: ``g_easting``
        - Tensor components:
            - ``g_ee``
            - ``g_nn``
            - ``g_zz``
            - ``g_en``
            - ``g_ez``
            - ``g_nz``

    coordinate_system : str (optional)
        Coordinate system of the coordinates of the computation points and the
        point masses.
        Available coordinates systems: ``cartesian``, ``spherical``.
        Default ``cartesian``.
    parallel : bool (optional)
        If True the computations will run in parallel using Numba built-in
        parallelization. If False, the forward model will run on a single core.
        Might be useful to disable parallelization if the forward model is run
        by an already parallelized workflow. Default to True.
    dtype : data-type (optional)
        Data type assigned to resulting gravitational field. Default to
        ``np.float64``.


    Returns
    -------
    result : array
        Gravitational field generated by the ``point_mass`` on the computation
        points defined in ``coordinates``.
        The potential is given in SI units, the accelerations in mGal and the
        Marussi tensor components in Eotvos.
    """
    # Sanity checks for coordinate_system
    check_coordinate_system(
        coordinate_system, valid_coord_systems=("cartesian", "spherical")
    )
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Prepare arrays to be passed to the jitted functions
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    points = tuple(np.atleast_1d(i).ravel() for i in points[:3])
    masses = np.atleast_1d(masses).ravel()
    # Sanity checks
    if masses.size != points[0].size:
        raise ValueError(
            "Number of elements in masses ({}) ".format(masses.size)
            + "mismatch the number of points ({})".format(points[0].size)
        )
    # Compute gravitational field
    kernel = get_kernel(coordinate_system, field)
    dispatcher(coordinate_system, parallel)(
        *coordinates, *points, masses, result, kernel
    )
    # Invert sign of gravity_u, gravity_eu, gravity_nu
    if field in ("g_z", "g_ez", "g_ze", "g_nz", "g_zn"):
        result *= -1
    # Convert to more convenient units
    if field in ("g_e", "g_n", "g_z"):
        result *= 1e5  # SI to mGal
    tensors = ("g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz", "g_ne", "g_ze", "g_zn")
    if field in tensors:
        result *= 1e9  # SI to Eotvos
    return result.reshape(cast.shape)


def dispatcher(coordinate_system, parallel):
    """
    Return the appropriate forward model function
    """
    dispatchers = {
        "cartesian": {
            True: point_mass_cartesian_parallel,
            False: point_mass_cartesian_serial,
        },
        "spherical": {
            True: point_mass_spherical_parallel,
            False: point_mass_spherical_serial,
        },
    }
    return dispatchers[coordinate_system][parallel]


def get_kernel(coordinate_system, field):
    """
    Return the appropriate kernel
    """
    kernels = {
        "cartesian": {
            "potential": gravity_pot,
            "g_z": gravity_u,
            "g_n": gravity_n,
            "g_e": gravity_e,
            # diagonal tensor components
            "g_ee": gravity_ee,
            "g_nn": gravity_nn,
            "g_zz": gravity_uu,
            # non-diagonal tensor components
            "g_en": gravity_en,
            "g_ez": gravity_eu,
            "g_nz": gravity_nu,
            "g_ne": gravity_en,
            "g_ze": gravity_eu,
            "g_zn": gravity_nu,
        },
        "spherical": {
            "potential": kernel_potential_spherical,
            "g_z": kernel_gravity_u_spherical,
            "g_n": None,
            "g_e": None,
        },
    }
    if field not in kernels[coordinate_system]:
        raise ValueError("Gravitational field '{}' not recognized".format(field))
    kernel = kernels[coordinate_system][field]
    if kernel is None:
        raise NotImplementedError
    return kernel


# ------------------------------------------
# Kernel functions for Spherical coordinates
# ------------------------------------------


@jit(nopython=True)
def kernel_potential_spherical(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Kernel function for potential gravitational field in spherical coordinates
    """
    distance, _, _ = distance_spherical_core(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    return 1 / distance * GRAVITATIONAL_CONST


#  Acceleration components
#  -------------------


@jit(nopython=True)
def kernel_gravity_u_spherical(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Kernel for upward component of gravitational acceleration

    Use spherical coordinates
    """
    distance, cospsi, _ = distance_spherical_core(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    delta_z = radius - radius_p * cospsi
    return -GRAVITATIONAL_CONST * delta_z / distance**3


def point_mass_cartesian(
    easting,
    northing,
    upward,
    easting_p,
    northing_p,
    upward_p,
    masses,
    out,
    forward_func,
):
    """
    Compute gravitational field of point masses in Cartesian coordinates

    Parameters
    ----------
    easting, northing, upward : 1d-arrays
        Coordinates of computation points in Cartesian coordinate system.
    easting_p, northing_p, upward_p : 1d-arrays
        Coordinates of point masses in Cartesian coordinate system.
    masses : 1d-array
        Mass of each point mass in SI units.
    out : 1d-array
        Array where the gravitational field on each computation point will be
        appended.
        It must have the same size of ``easting``, ``northing`` and ``upward``.
    forward_func : func
        forward_func function that will be used to compute the gravitational
        field on the computation points. It could be one of the forward
        modelling functions in :mod:`choclo.point`.
    """
    for l in prange(easting.size):
        for m in range(easting_p.size):
            out[l] += forward_func(
                easting[l],
                northing[l],
                upward[l],
                easting_p[m],
                northing_p[m],
                upward_p[m],
                masses[m],
            )


def point_mass_spherical(
    longitude, latitude, radius, longitude_p, latitude_p, radius_p, masses, out, kernel
):
    """
    Compute gravitational field of point masses in spherical coordinates

    Parameters
    ----------
    longitude, latitude, radius : 1d-arrays
        Coordinates of computation points in spherical geocentric coordinate
        system.
    longitude_p, latitude_p, radius_p : 1d-arrays
        Coordinates of point masses in spherical geocentric coordinate system.
    masses : 1d-array
        Mass of each point mass in SI units.
    out : 1d-array
        Array where the gravitational field on each computation point will be
        appended.
        It must have the same size of ``longitude``, ``latitude`` and
        ``radius``.
    kernel : func
        Kernel function that will be used to compute the gravitational field on
        the computation points.
    """
    # Compute quantities related to computation point
    longitude = np.radians(longitude)
    latitude = np.radians(latitude)
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    # Compute quantities related to point masses
    longitude_p = np.radians(longitude_p)
    latitude_p = np.radians(latitude_p)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    # Compute gravitational field
    for l in prange(longitude.size):
        for m in range(longitude_p.size):
            out[l] += masses[m] * kernel(
                longitude[l],
                cosphi[l],
                sinphi[l],
                radius[l],
                longitude_p[m],
                cosphi_p[m],
                sinphi_p[m],
                radius_p[m],
            )


# Define jitted versions of the forward modelling functions
point_mass_cartesian_serial = jit(nopython=True)(point_mass_cartesian)
point_mass_cartesian_parallel = jit(nopython=True, parallel=True)(point_mass_cartesian)
point_mass_spherical_serial = jit(nopython=True)(point_mass_spherical)
point_mass_spherical_parallel = jit(nopython=True, parallel=True)(point_mass_spherical)
