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

from .._spherical_harmonics.legendre import (
    assoc_legendre,
    assoc_legendre_deriv,
    assoc_legendre_full,
    assoc_legendre_schmidt,
    assoc_legendre_schmidt_deriv,
    assoc_legendre_full_deriv,
)


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


def legendre_derivative_analytical(x):
    "Analytical expressions for theta derivatives unnormalized Legendre functions"
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


def schmidt_normalization(max_degree):
    "Calculate the Schmidt normalization factor"
    s = np.full((max_degree + 1, max_degree + 1), np.nan)
    for n in range(max_degree + 1):
        for m in range(n + 1):
            if m == 0:
                s[n, m] = 1
            else:
                s[n, m] = np.sqrt(2 * math.factorial(n - m) / math.factorial(n + m))
    return s


def full_normalization(max_degree):
    "Calculate the full normalization factor"
    s = np.full((max_degree + 1, max_degree + 1), np.nan)
    for n in range(max_degree + 1):
        for m in range(n + 1):
            s[n, m] = np.sqrt((n + 0.5) * math.factorial(n - m) / math.factorial(n + m))
    return s


def test_assoc_legendre():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 180):
        x = np.cos(angle)
        p_analytical = legendre_analytical(x)
        max_degree = p_analytical.shape[0] - 1
        p = assoc_legendre(x, max_degree)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(p_analytical[n, m], p[n, m], atol=1e-10)
            # Make sure the upper diagonal is all NaNs
            for m in range(n + 1, max_degree + 1):
                assert np.isnan(p[n, m])


def test_assoc_legendre_schmidt():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 180):
        x = np.cos(angle)
        p_analytical_unnormalized = legendre_analytical(x)
        max_degree = p_analytical_unnormalized.shape[0] - 1
        s = schmidt_normalization(max_degree)
        p_analytical = s * p_analytical_unnormalized
        p = assoc_legendre_schmidt(x, max_degree)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(
                    p_analytical[n, m], p[n, m], atol=1e-10, err_msg=f"n={n}, m={m}"
                )
            # Make sure the upper diagonal is all NaNs
            for m in range(n + 1, max_degree + 1):
                assert np.isnan(p[n, m])


def test_assoc_legendre_full():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 180):
        x = np.cos(angle)
        p_analytical_unnormalized = legendre_analytical(x)
        max_degree = p_analytical_unnormalized.shape[0] - 1
        s = full_normalization(max_degree)
        p_analytical = s * p_analytical_unnormalized
        p = assoc_legendre_full(x, max_degree)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(
                    p_analytical[n, m], p[n, m], atol=1e-10, err_msg=f"n={n}, m={m}"
                )
            # Make sure the upper diagonal is all NaNs
            for m in range(n + 1, max_degree + 1):
                assert np.isnan(p[n, m])


def test_assoc_legengre_schmidt_identity():
    "Check Schmidt normalized functions against a known identity"
    # Higher degrees than this yield bad results
    max_degree = 600
    true_value = np.ones(max_degree + 1)
    for x in np.linspace(-1, 1, 100):
        p = assoc_legendre_schmidt(x, max_degree)
        p[np.isnan(p)] = 0
        np.testing.assert_allclose((p**2).sum(axis=1), true_value, atol=1e-12)


def test_assoc_legendre_deriv():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 360):
        x = np.cos(angle)
        p_analytical = legendre_derivative_analytical(x)
        max_degree = p_analytical.shape[0] - 1
        p = assoc_legendre(x, max_degree)
        dp = assoc_legendre_deriv(x, p)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(
                    p_analytical[n, m], dp[n, m], atol=1e-10, err_msg=f"n={n}, m={m}"
                )
            # Make sure the upper diagonal is all NaNs
            for m in range(n + 1, max_degree + 1):
                assert np.isnan(dp[n, m])


def test_assoc_legendre_schmidt_deriv():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 360):
        x = np.cos(angle)
        p_analytical_unnormalized = legendre_derivative_analytical(x)
        max_degree = p_analytical_unnormalized.shape[0] - 1
        s = schmidt_normalization(max_degree)
        p_analytical = s * p_analytical_unnormalized
        p = assoc_legendre_schmidt(x, max_degree)
        dp = assoc_legendre_schmidt_deriv(x, p)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(
                    p_analytical[n, m], dp[n, m], atol=1e-10, err_msg=f"n={n}, m={m}"
                )
            # Make sure the upper diagonal is all NaNs
            for m in range(n + 1, max_degree + 1):
                assert np.isnan(dp[n, m])


def test_assoc_legendre_full_deriv():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 360):
        x = np.cos(angle)
        p_analytical_unnormalized = legendre_derivative_analytical(x)
        max_degree = p_analytical_unnormalized.shape[0] - 1
        s = full_normalization(max_degree)
        p_analytical = s * p_analytical_unnormalized
        p = assoc_legendre_full(x, max_degree)
        dp = assoc_legendre_full_deriv(x, p)
        for n in range(max_degree + 1):
            for m in range(n + 1):
                np.testing.assert_allclose(
                    p_analytical[n, m], dp[n, m], atol=1e-10, err_msg=f"n={n}, m={m}"
                )
            # Make sure the upper diagonal is all NaNs
            for m in range(n + 1, max_degree + 1):
                assert np.isnan(dp[n, m])
