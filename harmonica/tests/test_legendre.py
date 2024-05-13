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
    assoc_legendre_full,
    assoc_legendre_schmidt,
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


def test_legendre_assoc_legendre():
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


def test_legendre_assoc_legendre_schmidt():
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


def test_legendre_assoc_legendre_full():
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
