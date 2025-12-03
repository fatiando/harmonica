# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import numpy.typing as npt
from scipy.special import ellipeinc, ellipkinc

from ..utils import get_rotation_matrix

# Relative tolerance for two ellipsoid semiaxes to be considered almost equal.
# E.g.: two semiaxes a and b are considered almost equal if:
# | a - b | <  max(a, b) * SEMIAXES_RTOL
SEMIAXES_RTOL = 1e-5


def get_semiaxes_rotation_matrix(ellipsoid):
    """
    Build extra rotation matrix to align semiaxes in decreasing order.

    Build a 90 degrees rotations matrix that goes from a local coordinate system where:

    - ``x`` points in the direction of ``a``,
    - ``y`` points in the direction of ``b``, and
    - ``z`` points in the direction of ``c``,

    where ``a >= b >= c``, to a *primed* local coordinate system where:

    - ``x'`` points in the direction of ``ellipsoid.a``,
    - ``y'`` points in the direction of ``ellipsoid.b``, and
    - ``z'`` points in the direction of ``ellipsoid.c``,

    and ``ellipsoid.a``, ``ellipsoid.b`` and ``ellipsoid.c`` have no particular order.
    The ``a``, ``b``, ``c`` are defined as:

    .. code::python

        a, b, c = sorted((ellipsoid.a, ellipsoid.b, ellipsoid.c), reverse=True)

    Parameters
    ----------
    ellipsoid : Ellipsoid
        Ellipsoid object.

    Returns
    -------
    rotation_matrix : (3, 3) np.ndarray
        Rotation matrix.

    Notes
    -----
    This matrix is not necessarily a permutation matrix, since it can contain -1.
    But it acts as a permutation matrix that ensures that ``x``, ``y``, ``z`` form
    a right-handed system.
    """
    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
    if a >= b >= c:
        return np.eye(3, dtype=int)

    if b >= a >= c:
        yaw, pitch, roll = 90, 0, 0
    elif c >= b >= a:
        yaw, pitch, roll = 0, 90, 0
    elif a >= c >= b:
        yaw, pitch, roll = 0, 0, 90
    elif b >= c >= a:
        yaw, pitch, roll = 90, 0, 90
    elif c >= a >= b:
        yaw, pitch, roll = 90, 90, 0
    else:
        raise ValueError()

    matrix = get_rotation_matrix(yaw, pitch, roll).astype(int)
    return matrix


def is_internal(x, y, z, a, b, c):
    """
    Check if a given point(s) is internal or external to the ellipsoid.

    Parameters
    ----------
    x, y, z : (n,) arrays or floats
        Coordinates of the observation point(s) in the local coordinate system.
    a, b, c : floats
        Ellipsoid's semiaxes lengths.

    Returns
    -------
    bool or (n,) array
    """
    return ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1


def is_almost_a_sphere(a: float, b: float, c: float) -> bool:
    """
    Check if a given ellipsoid approximates a sphere.

    Returns True if ellipsoid's semiaxes lengths are close enough to each other to be
    approximated by a sphere.

    Parameters
    ----------
    a, b, c: float
        Ellipsoid's semiaxes lenghts.

    Returns
    -------
    bool
    """
    # Exactly a sphere
    if a == b == c:
        return True

    # Prolate that is almost a sphere
    if b == c and np.abs(a - b) < SEMIAXES_RTOL * max(a, b):
        return True

    # Oblate that is almost a sphere
    if a == b and np.abs(b - c) < SEMIAXES_RTOL * max(b, c):
        return True

    # Triaxial that is almost a sphere
    if (  # noqa: SIM103
        a != b != c
        and np.abs(a - b) < SEMIAXES_RTOL * max(a, b)
        and np.abs(b - c) < SEMIAXES_RTOL * max(b, c)
    ):
        return True

    return False


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
    lambda_ : float or array-like
        The computed value(s) of the lambda parameter.

    """
    # Solve lambda for prolate and oblate ellipsoids
    if b == c:
        p0 = a**2 * b**2 - b**2 * x**2 - a**2 * (y**2 + z**2)
        p1 = a**2 + b**2 - x**2 - y**2 - z**2
        lambda_ = 0.5 * (np.sqrt(p1**2 - 4 * p0) - p1)

    # Solve lambda for triaxial ellipsoids
    else:
        p0 = (
            a**2 * b**2 * c**2
            - b**2 * c**2 * x**2
            - c**2 * a**2 * y**2
            - a**2 * b**2 * z**2
        )
        p1 = (
            a**2 * b**2
            + b**2 * c**2
            + c**2 * a**2
            - (b**2 + c**2) * x**2
            - (c**2 + a**2) * y**2
            - (a**2 + b**2) * z**2
        )
        p2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2
        p = p1 - (p2**2) / 3
        q = p0 - ((p1 * p2) / 3) + 2 * (p2 / 3) ** 3
        cos_theta = -q / (2 * np.sqrt((-p / 3) ** 3))

        # Clip the cos_theta to [-1, 1]. Due to floating point errors its value
        # could be slightly above 1 or slightly below -1.
        if isinstance(cos_theta, np.ndarray):
            # Run inplace to avoid allocating a new array.
            np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
        else:
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        lambda_ = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - p2 / 3

    return lambda_


def get_elliptical_integrals(
    a: float, b: float, c: float, lambda_: float | npt.NDArray
) -> tuple[float, float, float] | tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
    lambda_ : float or (n,) array
        The given lambda value for the point we are considering.

    Returns
    -------
    A, B, C : floats or tuple of (n,) arrays
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
        g1, g2, g3 = _get_elliptical_integrals_triaxial(a, b, c, lambda_)
    elif a > b and b == c:
        g1, g2, g3 = _get_elliptical_integrals_prolate(a, b, lambda_)
    elif a == b and b > c:
        g1, g2, g3 = _get_elliptical_integrals_oblate(b, c, lambda_)
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

    g1 = (2 / (sqrt_e**3)) * (log - sqrt_e / sqrt_l1)
    g2 = (1 / (sqrt_e**3)) * ((sqrt_e * sqrt_l1) / (b**2 + lmbda) - log)
    return g1, g2, g2


def _get_elliptical_integrals_oblate(b, c, lmbda):
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
            \frac{ 1 }{ (b^2 - c^2)^{\frac{3}{2}} }
            \left\{
                \arctan \left[ \sqrt{ \frac{b^2 - c^2}{c^2 + \lambda} } \right]
                -
                \frac{
                    \sqrt{ (b^2 - c^2) (c^2 + \lambda) }
                }{
                    b^2 + \lambda
                }
            \right\}

    .. math::

        C(\lambda) =
            \frac{ 2 }{ (b^2 - c^2)^{\frac{3}{2}} }
            \left\{
                \sqrt{ \frac{b^2 - c^2}{c^2 + \lambda} }
                -
                \arctan \left[ \sqrt{ \frac{b^2 - c^2}{c^2 + \lambda} } \right]
            \right\}


    and

    .. math::

        B(\lambda) = A(\lambda)

    .. important::

        These equations are modified versions of the one in Takahashi (2018), adapted to
        any oblate ellipsoid defined as: ``a = b > c``.

    """
    arctan = np.arctan(np.sqrt((b**2 - c**2) / (c**2 + lmbda)))
    g1 = (
        1
        / ((b**2 - c**2) ** (3 / 2))
        * (arctan - (np.sqrt((b**2 - c**2) * (c**2 + lmbda))) / (b**2 + lmbda))
    )
    g3 = (
        2
        / ((b**2 - c**2) ** (3 / 2))
        * ((np.sqrt((b**2 - c**2) / (c**2 + lmbda))) - arctan)
    )
    return g1, g1, g3


def get_derivatives_of_elliptical_integrals(
    a: float, b: float, c: float, lambda_: float | npt.NDArray
):
    r"""
    Compute derivatives of the elliptical integrals with respect to lambda.

    Compute the derivatives of the elliptical integrals :math:`A(\lambda)`,
    :math:`B(\lambda)`, and :math:`C(\lambda)` with respect to lambda.

    Parameters
    ----------
    a, b, c : floats
        Semi-axes lengths of the given ellipsoid.
    lambda_ : float
        The given lambda value for the point we are considering.

    Returns
    -------
    hx, hy, hz : tuple of floats
        The h values for the given observation point.
    """
    r = np.sqrt((a**2 + lambda_) * (b**2 + lambda_) * (c**2 + lambda_))
    hx, hy, hz = tuple(-1 / (e**2 + lambda_) / r for e in (a, b, c))
    return hx, hy, hz
