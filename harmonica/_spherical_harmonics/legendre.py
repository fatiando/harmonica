# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Calculation of associated Legendre functions and their derivatives.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def _rescale(u, max_degree, p):
    """Rescale Legendre functions to their original range"""
    rescale = 1e280
    for m in range(max_degree + 1):
        if m > 0:
            rescale *= u
        for n in range(m, max_degree + 1):
            p[n, m] *= rescale


@numba.jit(nopython=True)
def associated_legendre(x, max_degree, p):
    """
    Unnormalized associated Legendre functions up to a maximum degree.

    The functions :math:`P_n^m` are solutions to Legendre's equation. We
    calculate their values using the recursive relations defined in Alken
    (2022).

    .. note::

        This function does not include the Condon-Shortly phase.

    .. warning::

        Unnormalized functions should not be used beyond degree 13 because they
        will overflow 64-bit float range.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    """
    u = np.sqrt((1 - x) * (1 + x))
    p[0, 0] = 1
    for n in range(1, max_degree + 1):
        for m in range(n - 1):
            a_nm = (2 * n - 1) / (n - m)
            b_nm = -(n + m - 1) / (n - m)
            p[n, m] = a_nm * x * p[n - 1, m] + b_nm * p[n - 2, m]
        c_nm = 2 * n - 1
        p[n, n - 1] = c_nm * x * p[n - 1, n - 1]
        d_nm = 2 * n - 1
        p[n, n] = d_nm * u * p[n - 1, n - 1]


@numba.jit(nopython=True)
def associated_legendre_derivative(max_degree, p, dp):
    """
    Derivatives in theta of unnormalized associated Legendre functions.

    Calculates the derivative:

    .. math::

        \\dfrac{\\partial P_n^m}{\\partial \\theta}(\\cos \\theta)

    using the recursive relations defined in Alken (2022).

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument.

    .. note::

        This function does not include the Condon-Shortly phase.

    .. warning::

        Unnormalized functions should not be used beyond degree 13 because they
        will overflow 64-bit float range.

    Parameters
    ----------
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.ndarray
        A 2D array with the unnormalized associated Legendre functions
        calculated for :math:`\\cos\\theta`.
    dp : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    """
    dp[0, 0] = 0
    for n in range(1, max_degree + 1):
        a_nm = -1
        dp[n, 0] = a_nm * p[n, 1]
        for m in range(1, n):
            b_nm = 0.5 * (n + m) * (n - m + 1)
            c_nm = -0.5
            dp[n, m] = b_nm * p[n, m - 1] + c_nm * p[n, m + 1]
        d_nm = n
        dp[n, n] = d_nm * p[n, n - 1]


@numba.jit(nopython=True)
def associated_legendre_schmidt(x, max_degree, p):
    """
    Schmidt normalized associated Legendre functions up to maximum degree.

    The functions :math:`P_n^m` are solutions to Legendre's equation. The
    Schmidt normalization is applied to the functions to constrain their value
    range. This normalization if often used in geomagnetic field models. We
    calculate their values using the recursive relations defined in Alken
    (2022).

    .. note::

        This function does not include the Condon-Shortly phase.

    .. note::

        This function uses the scaling scheme of Holmes and Featherstone (2002)
        and produces accurate results until degree and order 2800.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    """
    u = np.sqrt((1 - x) * (1 + x))
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    # Use the Holmes and Featherstone (2002) scaling to compute scaled Pnm
    # All terms are scaled by the max float range 1e280
    p[0, 0] = 1 * 1e-280
    p[1, 0] = x * 1e-280
    # This equation doesn't have u because of the scaling
    p[1, 1] = p[0, 0]
    # Calculate the zero order terms first
    for n in range(2, max_degree + 1):
        a_n0 = (2 * n - 1) / (sqrt[n] ** 2)
        b_n0 = -((sqrt[n - 1] / sqrt[n]) ** 2)
        p[n, 0] = a_n0 * x * p[n - 1, 0] + b_n0 * p[n - 2, 0]
    # Now calculate the other terms
    for n in range(2, max_degree + 1):
        for m in range(1, n - 1):
            a_nm = (2 * n - 1) / (sqrt[n + m] * sqrt[n - m])
            b_nm = -sqrt[n + m - 1] * sqrt[n - m - 1] / (sqrt[n + m] * sqrt[n - m])
            p[n, m] = a_nm * x * p[n - 1, m] + b_nm * p[n - 2, m]
        c_nm = sqrt[2 * n - 1]
        p[n, n - 1] = c_nm * x * p[n - 1, n - 1]
        d_nm = sqrt[2 * n - 1] / sqrt[2 * n]
        # This equation doesn't have u because of the scaling
        p[n, n] = d_nm * p[n - 1, n - 1]
    # Now return everything to the original float range and rescale by u**m
    _rescale(u, max_degree, p)


@numba.jit(nopython=True)
def associated_legendre_schmidt_derivative(max_degree, p, dp):
    """
    Derivatives in theta of Schmidt normalized associated Legendre functions.

    Calculates the derivative:

    .. math::

        \\dfrac{\\partial P_n^m}{\\partial \\theta}(\\cos \\theta)

    using the recursive relations defined in Alken (2022).

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument.

    .. note::

        This function does not include the Condon-Shortly phase.

    .. note::

        First and second order derivatives are tested against the Legendre
        differential equation and produce accurate results until degree and
        order 2800.

    Parameters
    ----------
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.ndarray
        A 2D array with the Schmidt normalized associated Legendre functions
        calculated for :math:`\\cos\\theta`.
    dp : numpy.array
        An 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will
        be filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    """
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    dp[0, 0] = 0
    dp[1, 0] = -p[1, 1]
    dp[1, 1] = p[1, 0]
    for n in range(2, max_degree + 1):
        a_nm = -sqrt[n] * sqrt[n + 1] / sqrt[2]
        dp[n, 0] = a_nm * p[n, 1]
        b_nm = 0.5 * sqrt[n] * sqrt[n + 1] * sqrt[2]
        c_nm = -0.5 * sqrt[n + 2] * sqrt[n - 1]
        dp[n, 1] = b_nm * p[n, 0] + c_nm * p[n, 2]
        for m in range(2, n):
            b_nm = 0.5 * sqrt[n + m] * sqrt[n - m + 1]
            c_nm = -0.5 * sqrt[n + m + 1] * sqrt[n - m]
            dp[n, m] = b_nm * p[n, m - 1] + c_nm * p[n, m + 1]
        d_nm = 0.5 * sqrt[2] * sqrt[n]
        dp[n, n] = d_nm * p[n, n - 1]


@numba.jit(nopython=True)
def associated_legendre_full(x, max_degree, p):
    """
    Fully normalized associated Legendre functions up to maximum degree.

    The functions :math:`P_n^m` are solutions to Legendre's equation. The full
    normalization is applied to the functions to constrain their value range.
    This normalization if often used in gravity field models. We calculate
    their values using the recursive relations defined in Alken (2022) but with
    the definition of the normalization from Hofmann-Wellenhof and Moritz
    (2006).

    .. note::

        This function does not include the Condon-Shortly phase.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    Hofmann-Wellenhof, B., & Moritz, H. (2006). Physical Geodesy (2nd, corr.
      ed. 2006 edition ed.). Wien ; New York: Springer.
    """
    u = np.sqrt((1 - x) * (1 + x))
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    # Use the Holmes and Featherstone (2002) scaling to compute scaled Pnm
    # All terms are scaled by the max float range 1e280
    p[0, 0] = 1 * 1e-280
    p[1, 0] = x * sqrt[3] * 1e-280
    # This equation doesn't have u because of the scaling
    p[1, 1] = p[0, 0] * sqrt[3]
    # Calculate the zero order terms first
    for n in range(2, max_degree + 1):
        a_n0 = sqrt[2 * n - 1] * sqrt[2 * n + 1] / n
        b_n0 = -(n - 1) * sqrt[2 * n + 1] / (n * sqrt[2 * n - 3])
        p[n, 0] = a_n0 * x * p[n - 1, 0] + b_n0 * p[n - 2, 0]
    # Now calculate the other terms
    for n in range(2, max_degree + 1):
        for m in range(1, n - 1):
            a_nm = sqrt[2 * n + 1] * sqrt[2 * n - 1] / (sqrt[n + m] * sqrt[n - m])
            b_nm = (
                -sqrt[n + m - 1]
                * sqrt[n - m - 1]
                * sqrt[2 * n + 1]
                / (sqrt[n + m] * sqrt[n - m] * sqrt[2 * n - 3])
            )
            p[n, m] = a_nm * x * p[n - 1, m] + b_nm * p[n - 2, m]
        c_nm = sqrt[2 * n + 1]
        p[n, n - 1] = c_nm * x * p[n - 1, n - 1]
        d_nm = sqrt[2 * n + 1] / sqrt[2 * n]
        # This equation doesn't have u because of the scaling
        p[n, n] = d_nm * p[n - 1, n - 1]
    # Now return everything to the original float range and rescale by u**m
    _rescale(u, max_degree, p)


@numba.jit(nopython=True)
def associated_legendre_full_derivative(max_degree, p, dp):
    """
    Derivatives in theta of fully normalized associated Legendre functions.

    Calculates the derivative:

    .. math::

        \\dfrac{\\partial P_n^m}{\\partial \\theta}(\\cos \\theta)

    Uses the recursive relations defined in Alken (2022) but with the
    definition of the normalization from Hofmann-Wellenhof and Moritz (2006).

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument.

    .. note::

        This function does not include the Condon-Shortly phase.

    Parameters
    ----------
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.ndarray
        A 2D array with the fully normalized associated Legendre functions
        calculated for :math:`\\cos\\theta`.
    dp : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    Hofmann-Wellenhof, B., & Moritz, H. (2006). Physical Geodesy (2nd, corr.
      ed. 2006 edition ed.). Wien ; New York: Springer.
    """
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    dp[0, 0] = 0
    dp[1, 0] = -p[1, 1]
    dp[1, 1] = p[1, 0]
    for n in range(2, max_degree + 1):
        a_nm = -sqrt[n] * sqrt[n + 1] / sqrt[2]
        dp[n, 0] = a_nm * p[n, 1]
        b_n1 = 0.5 * sqrt[2] * sqrt[n + 1] * sqrt[n]
        c_n1 = -0.5 * sqrt[n + 2] * sqrt[n - 1]
        dp[n, 1] = b_n1 * p[n, 0] + c_n1 * p[n, 2]
        for m in range(2, n):
            b_nm = 0.5 * sqrt[n + m] * sqrt[n - m + 1]
            c_nm = -0.5 * sqrt[n + m + 1] * sqrt[n - m]
            dp[n, m] = b_nm * p[n, m - 1] + c_nm * p[n, m + 1]
        d_nm = 0.5 * sqrt[2] * sqrt[n]
        dp[n, n] = d_nm * p[n, n - 1]
