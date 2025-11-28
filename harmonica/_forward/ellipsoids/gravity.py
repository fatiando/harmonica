# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling of a gravity anomaly produced due to an ellipsoidal body.
"""

import warnings
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from choclo.constants import GRAVITATIONAL_CONST

from ...errors import NoPhysicalPropertyWarning
from ...typing import Coordinates, Ellipsoid
from .utils import (
    calculate_lambda,
    get_elliptical_integrals,
    get_permutation_matrix,
    is_almost_a_sphere,
    is_internal,
)


def ellipsoid_gravity(
    coordinates: Coordinates, ellipsoids: Iterable[Ellipsoid] | Ellipsoid
):
    r"""
    Forward model gravity fields of ellipsoids.

    Compute the gravity acceleration components for an ellipsoidal body at specified
    observation points.

    .. warning::

        The **vertical direction points upwards**, i.e. positive and negative
        values of ``upward`` represent points above and below the surface,
        respectively. But, ``g_z`` is the **downward component** of
        the gravitational acceleration so that positive density contrasts
        produce positive anomalies.

    .. important::

        The gravity acceleration components are returned in mGal
        (:math:`\text{m}/\text{s}^2`).

    Parameters
    ----------
    coordinates : list of array
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    ellipsoid : ellipsoid or list of ellipsoids
        Ellipsoidal body represented by an instance of
        :class:`harmonica.TriaxialEllipsoid`, :class:`harmonica.ProlateEllipsoid`,
        or :class:`harmonica.OblateEllipsoid`, or a list of them.

    Returns
    -------
    g_e, g_n, g_z : array
        Easting, northing and downward component of the gravity acceleration in mGal.

    References
    ----------
    [Clark1986]_
    [Takahashi2018]_

    For derivations of the equations, and methods used in this code.
    """
    # Cache broadcast of coordinates
    cast = np.broadcast(*coordinates)

    # Ravel coordinates into 1d arrays
    easting, northing, upward = tuple(np.atleast_1d(c).ravel() for c in coordinates)

    # Allocate arrays
    ge, gn, gu = tuple(np.zeros(easting.size) for _ in range(3))

    # Deal with the case of a single ellipsoid being passed
    if not isinstance(ellipsoids, Iterable):
        ellipsoids = [ellipsoids]

    for ellipsoid in ellipsoids:
        # Skip ellipsoid without density
        if ellipsoid.density is None:
            msg = (
                f"Ellipsoid {ellipsoid} doesn't have a density value. "
                "It will be skipped."
            )
            warnings.warn(msg, NoPhysicalPropertyWarning, stacklevel=2)
            continue

        # Translate observation points to coordinate system in center of the ellipsoid
        origin_e, origin_n, origin_u = ellipsoid.center
        coords_shifted = (easting - origin_e, northing - origin_n, upward - origin_u)

        # Get permutation matrix and order the semiaxes
        permutation_matrix = get_permutation_matrix(ellipsoid)
        a, b, c = sorted((ellipsoid.a, ellipsoid.b, ellipsoid.c), reverse=True)

        # Combine the rotation and the permutation matrices
        rotation = ellipsoid.rotation_matrix.T @ permutation_matrix

        # Rotate observation points
        x, y, z = rotation @ np.vstack(coords_shifted)

        # Calculate gravity components on local coordinate system
        gravity_ellipsoid = _compute_gravity_ellipsoid(
            x, y, z, a, b, c, ellipsoid.density
        )

        # project onto upward unit vector, axis U
        ge_i, gn_i, gu_i = rotation.T @ np.vstack(gravity_ellipsoid)

        # sum contributions from each ellipsoid
        ge += ge_i
        gn += gn_i
        gu += gu_i

    # Get g_z as the opposite of g_u
    gz = -gu

    # Reshape gravity arrays and convert to mGal
    ge, gn, gz = tuple(g.reshape(cast.shape) * 1e5 for g in (ge, gn, gz))
    return ge, gn, gz


def _compute_gravity_ellipsoid(
    x: npt.NDArray,
    y: npt.NDArray,
    z: npt.NDArray,
    a: float,
    b: float,
    c: float,
    density: float,
):
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
    # Mask internal points
    internal = is_internal(x, y, z, a, b, c)

    if is_almost_a_sphere(a, b, c):
        # Fallback to sphere equations which are simpler
        factor = -4 / 3 * np.pi * a**3 * GRAVITATIONAL_CONST * density
        gx, gy, gz = tuple(np.zeros_like(x) for _ in range(3))

        gx[internal] = factor * x[internal] / a**3
        gy[internal] = factor * y[internal] / a**3
        gz[internal] = factor * z[internal] / a**3

        r = np.sqrt(x[~internal] ** 2 + y[~internal] ** 2 + z[~internal] ** 2)
        gx[~internal] = factor * x[~internal] / r**3
        gy[~internal] = factor * y[~internal] / r**3
        gz[~internal] = factor * z[~internal] / r**3

        return gx, gy, gz

    # Compute lambda on all observation points:
    # Make it zero on internal points, calculate it for external points.
    lambda_ = np.zeros_like(x, dtype=np.float64)
    lambda_[~internal] = calculate_lambda(
        x[~internal], y[~internal], z[~internal], a, b, c
    )

    # Compute gx, gy, gz
    factor = -2 * np.pi * a * b * c * GRAVITATIONAL_CONST * density
    g1, g2, g3 = get_elliptical_integrals(a, b, c, lambda_)
    gx = factor * x * g1
    gy = factor * y * g2
    gz = factor * z * g3

    return gx, gy, gz
