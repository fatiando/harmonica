# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import numpy.typing as npt
from scipy.special import ellipeinc, ellipkinc

# Relative tolerance for two ellipsoid semiaxes to be considered almost equal.
# E.g.: two semiaxes a and b are considered almost equal if:
# | a - b | <  max(a, b) * SEMIAXES_RTOL
SEMIAXES_RTOL = 1e-5


def get_permutation_matrix(ellipsoid):
    """
    Get permutation matrix for the given ellipsoid.

    Build a permutation matrix that sorts the ellipsoid's semiaxes lengths
    ``ellipsoid.a``, ``ellipsoid.b``, ``ellipsoid.c`` into ``a``, ``b``, ``c`` values
    that satisfy that:

    - Triaxial: ``a > b > c``
    - Prolate: ``a > b == c``
    - Oblate: ``a < b == c``
    - Sphere: ``a == b == c``


    Parameters
    ----------
    ellipsoid : Ellipsoid
        Ellipsoid object.

    Returns
    -------
    permutation_matrix : (3, 3) np.ndarray
        Permutation matrix.

    Examples
    --------
    >>> from harmonica import Ellipsoid
    >>> ellipsoid = Ellipsoid(
    ...     a=43.0, b=50.0, c=83.0, yaw=0, pitch=0, roll=0, center=(0, 0, 0)
    ... )
    >>> get_permutation_matrix(ellipsoid)
    [[0 0 1]
     [0 1 0]
     [1 0 0]]

    >>> ellipsoid = Ellipsoid(
    ...     a=43.0, b=83.0, c=50.0, yaw=0, pitch=0, roll=0, center=(0, 0, 0)
    ... )
    >>> get_permutation_matrix(ellipsoid)
    [[0 1 0]
     [0 0 1]
     [1 0 0]]
    """
    # Sphere
    if ellipsoid.a == ellipsoid.b == ellipsoid.c:
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        permutation_matrix = np.eye(3)
        return a, b, c, permutation_matrix

    # Get permutation matrix
    permutation_matrix = _get_simple_permutation_matrix(
        ellipsoid.a, ellipsoid.b, ellipsoid.c
    )

    # Sort semiaxes so a >= b >= c
    c, b, a = sorted((ellipsoid.a, ellipsoid.b, ellipsoid.c))

    # Treat oblate ellipsoids
    if a == b:
        # Swap `a` with `c` so: a < b == c
        a, c = c, a
        permutation_matrix[0, :], permutation_matrix[2, :] = (
            permutation_matrix[2, :],
            permutation_matrix[0, :],
        )

    return permutation_matrix


def _get_simple_permutation_matrix(a, b, c):
    """
    Build permutation matrix for the given semiaxes.

    This permutation matrix sorts the semiaxes ``a``, ``b``, ``c`` in reverse order.

    Parameters
    ----------
    a, b, c : float
        Semiaxes lenghts.

    Returns
    -------
    permutation_matrix : (3, 3) np.ndarray
        Permutation matrix.
    """
    # Get sort indices
    semiaxes = np.array([a, b, c])
    argsort = np.argsort(semiaxes)
    # Build permutation matrix
    p_matrix = np.zeros((3, 3), dtype=np.int32)
    sorted_indices = np.array([2, 1, 0])
    p_matrix[sorted_indices, argsort] = 1
    return p_matrix


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

    Returns True if ellipsoid's semiaxes lenghts are close enough to each other to be
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

    # Prolate or oblate that is almost a sphere
    if b == c and np.abs(a - b) < SEMIAXES_RTOL * max(a, b):
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
    elif a < b and b == c:
        g1, g2, g3 = _get_elliptical_integrals_oblate(a, b, lambda_)
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
