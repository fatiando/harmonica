# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
from scipy.constants import mu_0
from scipy.spatial.transform import Rotation as rot
from scipy.special import ellipeinc, ellipkinc

import harmonica as hm


def _calculate_lambda(x, y, z, a, b, c):
    """
    Get the lambda ellipsoidal coordinate for a given ellipsoid and observation points.

    Calculate the value of lambda, the parameter defining surfaces in a
    confocal family of ellipsoids (i.e., the inflation/deflation parameter),
    for a given ellipsoid and observation point.

    Parameters
    ----------
    x : float or array
        X-coordinate(s) of the observation point(s) in the local coordinate
        system.
    y : float or array
        Y-coordinate(s) of the observation point(s).
    z : float or array
        Z-coordinate(s) of the observation point(s).
    a : float
        Semi-major axis of the ellipsoid along the x-direction.
    b : float
        Semi-major axis of the ellipsoid along the y-direction.
    c : float
        Semi-major axis of the ellipsoid along the z-direction.

    Returns
    -------
    lmbda : float or array-like
        The computed value(s) of the lambda parameter.

    """
    # compute lambda
    p_0 = (
        a**2 * b**2 * c**2
        - b**2 * c**2 * x**2
        - c**2 * a**2 * y**2
        - a**2 * b**2 * z**2
    )
    p_1 = (
        a**2 * b**2
        + b**2 * c**2
        + c**2 * a**2
        - (b**2 + c**2) * x**2
        - (c**2 + a**2) * y**2
        - (a**2 + b**2) * z**2
    )
    p_2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2

    p = p_1 - (p_2**2) / 3

    q = p_0 - ((p_1 * p_2) / 3) + 2 * (p_2 / 3) ** 3

    theta_internal = -q / (2 * np.sqrt((-p / 3) ** 3))

    # clip to remove floating point precision errors (as per testing)
    theta_internal_1 = np.clip(theta_internal, -1.0, 1.0)

    theta = np.arccos(theta_internal_1)

    lmbda = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - p_2 / 3

    # lmbda[inside_mask] = 0

    return lmbda


def get_elliptical_integrals(a, b, c, lmbda):
    r"""
    Compute elliptical integrals used in gravity and magnetic forward modelling.

    Compute the elliptical integrals :math:`A(\lambda)`, :math:`B(\lambda)`, and
    :math:`C(\lambda)` (Clark et al., 1986) for a given ellipsoid.

    .. note::

        These integrals are also called :math:`g_1`, :math:`g_2`, and :math:`g_3` in
        Takahashi et al. (2018).

    Parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.
    lmbda : float
        The given lmbda value for the point we are considering.

    Returns
    -------
    floats
        The elliptical integrals evaluated for the given ellipsoid and observation
        point.

    Notes
    -----
    The elliptical integrals :math:`A(\lambda)`, :math:`B(\lambda)`, and
    :math:`C(\lambda)` are given by (Clark et al., 1986, Takahashi et al. 2018):

    .. math::

        A(\lambda) =
            \int\limits_\lambda^\infty
            \frac{ \text{d}u }{ (a^2 + u) R(u) },

    .. math::

        B(\lambda) =
            \int\limits_\lambda^\infty
            \frac{ \text{d}u }{ (b^2 + u) R(u) },

    and

    .. math::

        C(\lambda) =
            \int\limits_\lambda^\infty
            \frac{ \text{d}u }{ (c^2 + u) R(u) },

    where :math:`R(u) = \sqrt{(a^2 + u) (b^2 + u) (c^2 + u)}`.

    The values of :math:`A(0)`, :math:`B(0)` and :math:`C(0)` are used to calculate
    fields inside the ellipsoid (where :math:`\lambda = 0`).

    Expressions of the elliptic integrals vary for each type of ellipsoid (triaxial,
    oblate and prolate).
    """
    if a > b > c:
        g1, g2, g3 = _get_elliptical_integrals_triaxial(a, b, c, lmbda)
    elif a > b and b == c:
        g1, g2, g3 = _get_elliptical_integrals_prolate(a, b, lmbda)
    elif a < b and b == c:
        g1, g2, g3 = _get_elliptical_integrals_oblate(a, b, lmbda)
    else:
        msg = f"Invalid semiaxis lenghts: a={a}, b={b}, c={c}."
        raise ValueError(msg)
    return g1, g2, g3


def _get_elliptical_integrals_triaxial(a, b, c, lmbda):
    r"""
    Compute elliptical integrals for a triaxial ellipsoid.

    Parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.
    lmbda : float
        The given lmbda value for the point we are considering.

    Returns
    -------
    floats
        The elliptical integrals evaluated for the given ellipsoid and observation
        point.

    Notes
    -----
    The elliptical integrals :math:`A(\lambda)`, :math:`B(\lambda)`, and
    :math:`C(\lambda)` for a triaxial ellipsoid (a > b > c) are given by
    (Clark et al., 1986, Takahashi et al. 2018):

    .. math::

        A(\lambda) =
            \frac{ 2 }{ (a^2 - b^2) \sqrt{a^2 - c^2} }
            \left[ F(\kappa, \phi) - E(\kappa, \phi) \right]

    .. math::

        B(\lambda) =
            \frac{ 2 \sqrt{a^2 - c^2} }{ (a^2 - b^2) (b^2 - c^2) }
            \left[
                E(\kappa, \phi)
                - \frac{b^2 - c^2}{a^2 - c^2} F(\kappa, \phi)
                - \frac{a^2 - b^2}{\sqrt{a^2 - c^2}}
                \sqrt{ \frac{c^2 + \lambda}{(a^2 + \lambda) (b^2 + \lambda)} }
            \right]

    and

    .. math::

        C(\lambda) =
            \frac{ -2 }{ (b^2 - c^2) \sqrt{a^2 - c^2} } E(\kappa, \phi)
            +
            \frac{2}{b^2 - c^2}
            \sqrt{ \frac{b^2 + \lambda}{(a^2 + \lambda) (c^2 + \lambda)} }

    where

    .. math::

        \sin \phi = \sqrt{\frac{a^2 - c^2}{a^2 + \lambda}}
        \quad
        (0 \le \theta' \le \pi/2),

    .. math::

        \kappa = \sqrt{\frac{a^2 - b^2}{a^2 - c^2}},

    and :math:`E(\kappa, \phi)` and :math:`F(\kappa, \phi)` are Legendre's normal
    elliptic integrals of the first and second kind, respectively.

    .. note::

        Note that the equation for :math:`C(\lambda)` includes a minus sign in the term
        that includes the :math:`E(\kappa, \phi)` integral that is missing in eq. 40
        from Takahashi et al. (2018).

    """
    # Compute phi and kappa
    int_arcsin = np.sqrt((a**2 - c**2) / (a**2 + lmbda))
    phi = np.arcsin(int_arcsin)
    k = (a**2 - b**2) / (a**2 - c**2)

    # Cache values of E(theta, k) and F(theta, k) so we compute them only once
    ellipk = ellipkinc(phi, k)
    ellipe = ellipeinc(phi, k)

    # A(lambda)
    g1 = (2 / ((a**2 - b**2) * (a**2 - c**2) ** 0.5)) * (ellipk - ellipe)

    # B(lambda)
    g2_multiplier = (2 * np.sqrt(a**2 - c**2)) / ((a**2 - b**2) * (b**2 - c**2))
    g2_elliptics = ellipe - ((b**2 - c**2) / (a**2 - c**2)) * ellipk
    g2_last_term = ((a**2 - b**2) / np.sqrt(a**2 - c**2)) * np.sqrt(
        (c**2 + lmbda) / ((a**2 + lmbda) * (b**2 + lmbda))
    )
    g2 = g2_multiplier * (g2_elliptics - g2_last_term)

    # C(lambda)
    # Term with the E(k, theta) must have a minus sign
    # (the minus sign is missing in Takahashi (2018)).
    g3_term_1 = -(2 / ((b**2 - c**2) * np.sqrt(a**2 - c**2))) * ellipe
    g3_term_2 = (2 / (b**2 - c**2)) * np.sqrt(
        (b**2 + lmbda) / ((a**2 + lmbda) * (c**2 + lmbda))
    )
    g3 = g3_term_1 + g3_term_2

    return g1, g2, g3


def _get_elliptical_integrals_prolate(a, b, lmbda):
    r"""
    Compute elliptical integrals for a prolate ellipsoid.

    Parameters
    ----------
    a, b : floats
        Semiaxis lengths of the given ellipsoid.
    lmbda : float
        The given lmbda value for the point we are considering.

    Returns
    -------
    floats
        The elliptical integrals evaluated for the given ellipsoid and observation
        point.

    Notes
    -----
    The elliptical integrals :math:`A(\lambda)`, :math:`B(\lambda)`, and
    :math:`C(\lambda)` for a prolate ellipsoid (a > b = c) are given by
    (Clark et al., 1986, Takahashi et al. 2018):

    .. math::

        A(\lambda) =
            \frac{ 2 }{ (a^2 - b^2)^{\frac{3}{2}} }
            \left\{
                \ln
                \left[
                    \frac{
                      \sqrt{a^2 - b^2} + \sqrt{a^2 + \lambda}
                    }{
                      \sqrt{b^2 + \lambda}
                    }
                \right]
                - \sqrt{\frac{a^2 - b^2}{a^2 + \lambda}}
            \right\}

    .. math::

        B(\lambda) =
            \frac{ 1 }{ (a^2 - b^2)^{\frac{3}{2}} }
            \left\{
                \frac{
                    \sqrt{(a^2 - b^2)(a^2 + \lambda)}
                }{
                    b^2 + \lambda
                }
                - \ln
                \left[
                    \frac{
                      \sqrt{a^2 - b^2} + \sqrt{a^2 + \lambda}
                    }{
                      \sqrt{b^2 + \lambda}
                    }
                \right]
            \right\}

    and

    .. math::

        C(\lambda) = B(\lambda)

    """
    # Cache some reused variables
    e2 = a**2 - b**2
    sqrt_e = np.sqrt(e2)
    sqrt_l1 = np.sqrt(a**2 + lmbda)
    sqrt_l2 = np.sqrt(b**2 + lmbda)
    log = np.log((sqrt_e + sqrt_l1) / sqrt_l2)

    g1 = (2 / (e2 ** (3 / 2))) * (log - sqrt_e / sqrt_l1)
    g2 = (1 / (e2 ** (3 / 2))) * ((sqrt_e * sqrt_l1) / (b**2 + lmbda) - log)
    return g1, g2, g2


def _get_elliptical_integrals_oblate(a, b, lmbda):
    r"""
    Compute elliptical integrals for a oblate ellipsoid.

    Parameters
    ----------
    a, b : floats
        Semiaxis lengths of the given ellipsoid.
    lmbda : float
        The given lmbda value for the point we are considering.

    Returns
    -------
    floats
        The elliptical integrals evaluated for the given ellipsoid and observation
        point.

    Notes
    -----
    The elliptical integrals :math:`A(\lambda)`, :math:`B(\lambda)`, and
    :math:`C(\lambda)` for a prolate ellipsoid (a < b = c) are given by
    (Clark et al., 1986, Takahashi et al. 2018):

    .. math::

        A(\lambda) =
            \frac{ 2 }{ (b^2 - a^2)^{\frac{3}{2}} }
            \left\{
                \sqrt{ \frac{b^2 - a^2}{a^2 + \lambda} }
                -
                \arctan \left[ \sqrt{ \frac{b^2 - a^2}{a^2 + \lambda} } \right]
            \right\}

    .. math::

        B(\lambda) =
            \frac{ 1 }{ (b^2 - a^2)^{\frac{3}{2}} }
            \left\{
                \arctan \left[ \sqrt{ \frac{b^2 - a^2}{a^2 + \lambda} } \right]
                -
                \frac{
                    \sqrt{ (b^2 - a^2) (a^2 + \lambda) }
                }{
                    b^2 + \lambda
                }
            \right\}

    and

    .. math::

        C(\lambda) = B(\lambda)

    """
    arctan = np.arctan(np.sqrt((b**2 - a**2) / (a**2 + lmbda)))
    g1 = (
        2
        / ((b**2 - a**2) ** (3 / 2))
        * ((np.sqrt((b**2 - a**2) / (a**2 + lmbda))) - arctan)
    )
    g2 = (
        1
        / ((b**2 - a**2) ** (3 / 2))
        * (arctan - (np.sqrt((b**2 - a**2) * (a**2 + lmbda))) / (b**2 + lmbda))
    )
    return g1, g2, g2


def _get_v_as_euler(yaw, pitch, roll):
    """
    Build rotation matrix from yaw, pitch and roll angles.

    Generate a rotation matrix (V) from Tait-Bryan angles: yaw, pitch, and roll.

    Parameters
    ----------
    yaw : float
        Rotation about the vertical (z) axis, in degrees.
    pitch : float
        Rotation about the northing (y) axis, in degrees.
    roll : float
        Rotation about the easting (x) axis, in degrees.

    Returns
    -------
    V : (3, 3) array
        Rotation matrix that transforms coordinates from the local ellipsoid-aligned
        frame to the global coordinate system.

    Notes
    -----
    The rotations are applied in the following order: (ZŶX).
    Yaw (Z) and roll (X) rotations are done using the right-hand rule. Rotations for the
    pitch (Ŷ) are carried out in the opposite direction, so positive pitch _lifts_ the
    easting axis.
    """
    # using scipy rotation package
    # this produces the local to global rotation matrix (or what would be
    # defined as r.T from global to local)
    m = rot.from_euler("zyx", [yaw, -pitch, roll], degrees=True)
    v = m.as_matrix()

    return v


# TODO: Remove this function, it's not being used
def _global_to_local(northing, easting, extra_coords, depth, v):
    """
    Convert observation points from global coordinates (Northing, Easting,
    Height)
    to local ellipsoid-aligned coordinates (x, y, z).

    Parameters
    ----------
    northing : array_like
        Northing (Y) coordinates in the global system.

    easting : array_like
        Easting (X) coordinates in the global system.

    extra_coords : array_like
        Height or vertical offset above the surface (commonly from
                                                     `vd.grid_coordinates`).

    depth : float
        Depth of the ellipsoid's center below the surface (positive downward).

    V : ndarray of shape (3, 3)
        Rotation matrix used to transform from global to local coordinates.

    Returns
    -------
    x, y, z : ndarray
        Coordinates of the observation points in the local ellipsoid-aligned
        frame.

    Notes
    -----
    Needs to handle translation component.

    """
    x = np.ones(northing.shape)
    y = np.ones(northing.shape)
    z = np.ones(northing.shape)
    local_coords = [x, y, z]

    # calculate local_coords for each x, y, z point
    for i in range(len(local_coords)):
        local_coords[i] = (
            northing * v[i][0] + easting * v[i][1] - (depth - extra_coords) * v[i][2]
        )

    return local_coords


# TODO: Remove this function, it's not being used
def _generate_basic_ellipsoid(a, b, c):
    """
    Generate the surface of an ellipsoid using spherical angles for 3D
    plotting.
    This function is seperate from gravity calculations and is purely for
    visualisation of 3D ellipsoids.

    Parameters
    ----------
    a, b, c : float
        Semiaxis lengths of the ellipsoid along the x, y, and z axes,
        respectively.

    Returns
    -------
    x1, y1, z1 : ndarray
        Arrays representing the ellipsoid surface coordinates in 3D space,
        computed
        from spherical angles. T

    """
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # np.outer is the outer product of the two arrays (ellipsoid surfce)
    x1 = a * np.outer(np.cos(u), np.sin(v))
    y1 = b * np.outer(np.sin(u), np.sin(v))
    z1 = c * np.outer(np.ones_like(u), np.cos(v))

    return x1, y1, z1


# TODO: move this function to test/utils.py
def _sphere_magnetic(coordinates, radius, center, magnetization):
    """
    Compute the magnetic field generated by a sphere with constant
    magnetization.

    The field inside the sphere is uniform. Outside the sphere it's equivalent
    to the dipole field.

    Parameters
    ----------
    coordinates : tuple of floats or arrays
        Coordinates of the observation points in the following order:
        easting, northing, upward.
    radius : float
        Radius of the sphere.
    center : tuple of floats
        Coordinates of the center of the sphere in the same order as the
        ``coordinates``.
    magnetization : tuple of floats
        Magnetization vector of the sphere.

    Returns
    -------
    be, bn, bu : arrays
        Three components of the magnetic field on the observation points in nT.
    """
    # Ravel the coordinates into 1d arrays
    cast = np.broadcast(*coordinates)
    easting, northing, upward = tuple(np.asarray(c).ravel() for c in coordinates)

    # Get the coordinates of the observation points in a coordinate system
    # located in the center of the sphere
    x, y, z = (easting - center[0], northing - center[1], upward - center[2])

    # Allocate arrays for the resulting magnetic field
    be, bn, bu = tuple(np.zeros(cast.size) for _ in range(3))

    # Compute the internal magnetic field in nT.
    magnetization = np.asarray(magnetization)
    inside = (x**2 + y**2 + z**2) <= radius**2
    be[inside] += 2 / 3 * mu_0 * magnetization[0] * 1e9
    bn[inside] += 2 / 3 * mu_0 * magnetization[1] * 1e9
    bu[inside] += 2 / 3 * mu_0 * magnetization[2] * 1e9

    # Compute the external magnetic field (a dipole field) in nT.
    # Get the magnetic moment of the equivalent dipole
    mag_moment = 4 / 3 * np.pi * radius**3 * magnetization
    be_out, bn_out, bu_out = hm.dipole_magnetic(
        (easting[~inside], northing[~inside], upward[~inside]),
        center,
        mag_moment,
        field="b",
    )
    be[~inside] = be_out
    bn[~inside] = bn_out
    bu[~inside] = bu_out

    be, bn, bu = tuple(b.reshape(cast.shape) for b in (be, bn, bu))
    return be, bn, bu
