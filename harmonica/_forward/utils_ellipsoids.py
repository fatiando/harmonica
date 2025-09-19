# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
from scipy.spatial.transform import Rotation as rot
from scipy.special import ellipeinc, ellipkinc


def calculate_lambda(x, y, z, a, b, c):
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
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The given lambda value for the point we are considering.

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
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The given lambda value for the point we are considering.

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
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The given lambda value for the point we are considering.

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
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The given lambda value for the point we are considering.

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


def get_rotation_matrix(yaw, pitch, roll):
    """
    Build rotation matrix from yaw, pitch and roll angles.

    Generate a rotation matrix (V) from Tait-Bryan intrinsic angles:
    yaw, pitch, and roll.

    Parameters
    ----------
    yaw : float
        Rotation about the vertical (Z) axis, in degrees.
    pitch : float
        Rotation about the northing (Y) axis, in degrees.
    roll : float
        Rotation about the easting (X) axis, in degrees.

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

    This rotation matrix allows to apply rotations from the local coordinate system of
    the ellipsoid into the global coordinate system (easting, northing, upward).
    """
    # using scipy rotation package
    # this produces the local to global rotation matrix (or what would be
    # defined as r.T from global to local)
    # Use capitalized axes for intrinsic rotations.
    m = rot.from_euler("ZYX", [yaw, -pitch, roll], degrees=True)
    v = m.as_matrix()

    return v


def get_derivatives_of_elliptical_integrals(a, b, c, lmbda):
    r"""
    Compute derivatives of the elliptical integrals with respect to lambda.

    Compute the derivatives of the elliptical integrals :math:`A(\lambda)`,
    :math:`B(\lambda)`, and :math:`C(\lambda)` with respect to lambda.

    Parameters
    ----------
    a, b, c : floats
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The given lambda value for the point we are considering.

    Returns
    -------
    hx, hy, hz : tuple of floats
        The h values for the given observation point.
    """
    r = np.sqrt((a**2 + lmbda) * (b**2 + lmbda) * (c**2 + lmbda))
    hx, hy, hz = tuple(-1 / (e**2 + lmbda) / r for e in (a, b, c))
    return hx, hy, hz
