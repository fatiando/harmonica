# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the associated Legendre function calculations.
"""
import math

import numpy as np
import pytest

from .._spherical_harmonics.legendre import (
    associated_legendre,
    associated_legendre_derivative,
    associated_legendre_full,
    associated_legendre_full_derivative,
    associated_legendre_schmidt,
    associated_legendre_schmidt_derivative,
)
from .utils import run_only_with_numba


def legendre_analytical(x):
    "Analytical expressions for unnormalized Legendre functions"
    max_degree = 4
    p = np.full((max_degree + 1, max_degree + 1), np.nan)
    p[0, 0] = 1
    p[1, 0] = x
    p[1, 1] = np.sqrt(1 - x**2)
    p[2, 0] = 1 / 2 * (3 * x**2 - 1)
    p[2, 1] = 3 * x * np.sqrt(1 - x**2)
    p[2, 2] = 3 * (1 - x**2)
    p[3, 0] = 1 / 2 * (5 * x**3 - 3 * x)
    p[3, 1] = -3 / 2 * (1 - 5 * x**2) * np.sqrt(1 - x**2)
    p[3, 2] = 15 * x * (1 - x**2)
    p[3, 3] = 15 * np.sqrt(1 - x**2) ** 3
    p[4, 0] = 1 / 8 * (35 * x**4 - 30 * x**2 + 3)
    p[4, 1] = 5 / 2 * (7 * x**3 - 3 * x) * np.sqrt(1 - x**2)
    p[4, 2] = 15 / 2 * (7 * x**2 - 1) * (1 - x**2)
    p[4, 3] = 105 * x * np.sqrt(1 - x**2) ** 3
    p[4, 4] = 105 * np.sqrt(1 - x**2) ** 4
    return p


@pytest.mark.use_numba
def legendre_derivative_analytical(x):
    """
    Analytical expressions for theta derivatives of unnormalized Legendre
    functions
    """
    max_degree = 4
    dp = np.full((max_degree + 1, max_degree + 1), np.nan)
    cos = x
    sin = np.sqrt(1 - x**2)
    dp[0, 0] = 0
    dp[1, 0] = -sin
    dp[1, 1] = cos
    dp[2, 0] = -3 * cos * sin
    dp[2, 1] = 3 * (cos**2 - sin**2)
    dp[2, 2] = 6 * sin * cos
    dp[3, 0] = (3 * sin - 15 * cos**2 * sin) / 2
    dp[3, 1] = 3 / 2 * (5 * cos**3 - 10 * cos * sin**2 - cos)
    dp[3, 2] = 30 * cos**2 * sin - 15 * sin**3
    dp[3, 3] = 45 * sin**2 * cos
    dp[4, 0] = (15 * cos - 35 * cos**3) * sin / 2
    dp[4, 1] = (
        35 * cos**4 - 15 * cos**2 + 15 * sin**2 - 105 * cos**2 * sin**2
    ) / 2
    dp[4, 2] = 105 * sin * cos**3 - 105 * cos * sin**3 - 15 * sin * cos
    dp[4, 3] = 315 * cos**2 * sin**2 - 105 * sin**4
    dp[4, 4] = 420 * sin**3 * cos
    return dp


def schmidt_normalization(p):
    "Multiply by the Schmidt normalization factor"
    max_degree = p.shape[0] - 1
    for n in range(max_degree + 1):
        for m in range(n + 1):
            if m > 0:
                p[n, m] *= np.sqrt(2 * math.factorial(n - m) / math.factorial(n + m))


def full_normalization(p):
    "Multiply by the full normalization factor"
    max_degree = p.shape[0] - 1
    for n in range(max_degree + 1):
        for m in range(n + 1):
            if m == 0:
                p[n, m] *= np.sqrt(2 * n + 1)
            else:
                p[n, m] *= np.sqrt(
                    2 * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m)
                )


@pytest.mark.use_numba
@pytest.mark.parametrize(
    "func,norm",
    (
        (associated_legendre, None),
        (associated_legendre_schmidt, schmidt_normalization),
        (associated_legendre_full, full_normalization),
    ),
    ids=["unnormalized", "schmidt", "full"],
)
def test_associated_legendre_function_analytical(func, norm):
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 360):
        x = np.cos(angle)
        # Analytical expression
        p_analytical = legendre_analytical(x)
        max_degree = p_analytical.shape[0] - 1
        if norm is not None:
            norm(p_analytical)
        # Numerical calculation
        p = np.empty((max_degree + 1, max_degree + 1))
        func(x, max_degree, p)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(p_analytical[n, m], p[n, m], atol=1e-10)


@pytest.mark.use_numba
@pytest.mark.parametrize(
    "func,deriv,norm",
    (
        (associated_legendre, associated_legendre_derivative, None),
        (
            associated_legendre_schmidt,
            associated_legendre_schmidt_derivative,
            schmidt_normalization,
        ),
        (
            associated_legendre_full,
            associated_legendre_full_derivative,
            full_normalization,
        ),
    ),
    ids=["unnormalized", "schmidt", "full"],
)
def test_associated_legendre_function_derivative_analytical(func, norm, deriv):
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 360):
        x = np.cos(angle)
        # Analytical expression
        dp_analytical = legendre_derivative_analytical(x)
        max_degree = dp_analytical.shape[0] - 1
        if norm is not None:
            norm(dp_analytical)
        # Numerical calculation
        p = np.empty((max_degree + 1, max_degree + 1))
        dp = np.empty((max_degree + 1, max_degree + 1))
        func(x, max_degree, p)
        deriv(max_degree, p, dp)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(
                    dp_analytical[n, m], dp[n, m], atol=1e-10, err_msg=f"n={n}, m={m}"
                )


class BaseSchmidt:
    """
    Base class to run tests using Schmidt identity.

    Child classes of this one need to define a ``max_degree`` attribute.
    Degrees higher than 2800 lead to bad results.
    """

    def test_associated_legengre_function_schmidt_identity(self):
        "Check Schmidt normalized functions against a known identity"
        # The sum of the coefs squared for a degree should be 1
        true_value = np.ones(self.max_degree + 1)
        p = np.zeros((self.max_degree + 1, self.max_degree + 1))
        for x in np.linspace(-1, 1, 50):
            associated_legendre_schmidt(x, self.max_degree, p)
            np.testing.assert_allclose(
                (p**2).sum(axis=1), true_value, atol=1e-10, rtol=0
            )

    # Not testing unnormalized ones because they only work until a very
    # low degree
    @pytest.mark.parametrize(
        "func,deriv",
        (
            (associated_legendre_schmidt, associated_legendre_schmidt_derivative),
            (associated_legendre_full, associated_legendre_full_derivative),
        ),
        ids=["schmidt", "full"],
    )
    def test_associated_legengre_function_legendre_equation(self, func, deriv):
        "Check functions and derivatives against the Legendre equation"
        max_degree = self.max_degree
        # Legendre equation should result in 0
        true_value = np.zeros((max_degree + 1, max_degree + 1))
        p = np.zeros((max_degree + 1, max_degree + 1))
        dp = np.zeros((max_degree + 1, max_degree + 1))
        dp2 = np.zeros((max_degree + 1, max_degree + 1))
        index = np.arange(max_degree + 1).reshape((max_degree + 1, 1))
        n = np.repeat(index, max_degree + 1, axis=1)
        m = np.repeat(index.T, max_degree + 1, axis=0)
        for angle in np.linspace(0.001, np.pi - 0.001, 10):
            cos = np.cos(angle)
            sin = np.sin(angle)
            func(cos, max_degree, p)
            deriv(max_degree, p, dp)
            deriv(max_degree, dp, dp2)
            legendre = sin * dp2 + cos * dp + (sin * n * (n + 1) - m**2 / sin) * p
            np.testing.assert_allclose(
                legendre, true_value, atol=1e-5, rtol=0, err_msg=f"angle={angle}"
            )


@run_only_with_numba
class TestSchmidtHighDegree(BaseSchmidt):
    max_degree = 2800


class TestSchmidtLowDegree(BaseSchmidt):
    max_degree = 30
