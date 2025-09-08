# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling of a gravity anomaly produced due to an ellipsoidal body.
"""

from collections.abc import Iterable

import numpy as np
from choclo.constants import GRAVITATIONAL_CONST

from .utils_ellipsoids import (
    _calculate_lambda,
    get_rotation_matrix,
    get_elliptical_integrals,
)


def ellipsoid_gravity(coordinates, ellipsoids, density, field="g"):
    r"""
    Forward model gravity fields of ellipsoids.

    Compute the gravity acceleration components for an ellipsoidal body at specified
    observation points.

    .. warning::

        The **vertical direction points upwards**, i.e. positive and negative
        values of ``upward`` represent points above and below the surface,
        respectively. But ``g_z`` field returns the **downward component** of
        the gravitational acceleration so that positive density contrasts
        produce positive anomalies.

    .. important::

        The gravity acceleration components are returned in mGal
        (:math:`\text{m}/\text{s}^2`).

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    ellipsoid : ellipsoid or list of ellipsoids
        Ellipsoidal body represented by an instance of
        :class:`harmonica.TriaxialEllipsoid`, :class:`harmonica.ProlateEllipsoid`, or
        :class:`harmonica.OblateEllipsoid`, or a list of them.
    density : float, list of floats or array
        List or array containing the density of each ellipsoid in kg/m^3.
    field : {"g", "g_e", "g_n", "g_z"}, optional
        Desired field that want to be computed.
        If "g_e", "g_n", "g_z" the function will return the easting, northing
        or downward gravity acceleration component, respectively.
        If "g", the function will return a tuple with the three components.
        Default to "g".

    Returns
    -------
    g_e, g_n, g_z: arrays
        Easting, northing and downward component of the gravity acceleration.
        Or a single one if ``field`` is "g_e", "g_n" or "g_z".

    References
    ----------
    Clark, S. A., et al. (1986), "Magnetic and gravity anomalies of a triaxial
    ellipsoid"
    Takahashi, Y., et al. (2018), "Magnetic modelling of ellipsoidal bodies"

    For derivations of the equations, and methods used in this code.
    """
    if field not in ("g", "g_e", "g_n", "g_z"):
        msg = f"Invalid field '{field}'. Choose from 'g', 'g_e', 'g_n', or 'g_z'."
        raise ValueError(msg)

    # Cache broadcast of coordinates
    cast = np.broadcast(*coordinates)

    # Ravel coordinates into 1d arrays
    easting, northing, upward = tuple(np.atleast_1d(c).ravel() for c in coordinates)

    # Allocate arrays
    ge, gn, gu = tuple(np.zeros(easting.size) for _ in range(3))

    # deal with the case of a single ellipsoid being passed
    if not isinstance(ellipsoids, Iterable):
        ellipsoids = [ellipsoids]
    if not isinstance(density, Iterable):
        density = [density]

    for ellipsoid, rho in zip(ellipsoids, density, strict=True):
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        origin_e, origin_n, origin_u = ellipsoid.centre

        # Translate observation points to coordinate system in center of the ellipsoid
        obs_points = np.vstack(
            (easting - origin_e, northing - origin_n, upward - origin_u)
        )

        # create rotation matrix
        r = get_rotation_matrix(yaw, pitch, roll)

        # rotate observation points
        rotated_points = r.T @ obs_points
        x, y, z = tuple(c for c in rotated_points)

        # calculate gravity component for the rotated points
        gx, gy, gz = _compute_gravity_ellipsoid(x, y, z, a, b, c, rho)
        gravity = np.vstack((gx, gy, gz))

        # project onto upward unit vector, axis U
        g_projected = r @ gravity
        ge_i, gn_i, gu_i = tuple(c for c in g_projected)

        # sum contributions from each ellipsoid
        ge += ge_i
        gn += gn_i
        gu += gu_i

    # Get gz
    gz = -gu

    # Reshape gravity arrays and convert to mGal
    ge, gn, gz = tuple(g.reshape(cast.shape) * 1e5 for g in (ge, gn, gz))

    if field == "g":
        return (ge, gn, gz)

    fields = {"g_e": ge, "g_n": gn, "g_z": gz}
    return fields[field]


def _compute_gravity_ellipsoid(x, y, z, a, b, c, density):
    """
    Compute gravity acceleration for an ellipsoid on a set of observation points.

    The observation points can either be internal or external.

    Parameters
    ----------
    x, y, z : arrays
        Observation coordinates in the local ellipsoid reference frame.
    a, b, c : floats
        Semiaxis lengths of the ellipsoid. Must conform to the constraints of
        the chosen ellipsoid type.
    density : float
        Density of the ellipsoidal body in kg/m³.

    Returns
    -------
    gx, gy, gz : arrays
        Gravity acceleration components in the local coordinate system for the
        ellipsoid. Accelerations are given in SI units (m/s^2).
    """
    # Compute lambda for all observation points
    lmbda = _calculate_lambda(x, y, z, a, b, c)

    # Clip lambda to zero for internal points
    inside = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1
    lmbda[inside] = 0

    # Compute gx, gy, gz
    factor = -2 * np.pi * a * b * c * GRAVITATIONAL_CONST * density
    g1, g2, g3 = get_elliptical_integrals(a, b, c, lmbda)
    gx = factor * x * g1
    gy = factor * y * g2
    gz = factor * z * g3

    return gx, gy, gz
