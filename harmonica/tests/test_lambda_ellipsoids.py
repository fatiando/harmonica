# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import pytest

from .._forward.utils_ellipsoids import calculate_lambda


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [(6.0, 5.0, 5.0), (4.0, 5.0, 5.0), (6.0, 5.0, 4.0)],
    ids=["prolate", "oblate", "triaxial"],
)
def test_lambda(a, b, c):
    """
    Test that lambda fits characteristic equation on external points.
    """
    # Build a 3D grid of observation points
    x, y, z = np.meshgrid(*[np.linspace(-10, 10, 41) for _ in range(3)])

    # Filter out only the external points to the ellipsoid
    internal = x**2 / a**2 + y**2 / b**2 + z**2 / c**2 < 1
    x, y, z = x[~internal], y[~internal], z[~internal]

    # Calculate lambda
    lambda_ = calculate_lambda(x, y, z, a, b, c)

    # Check if all lambda parameters are greater than -c**2
    assert (lambda_ > -(c**2)).all()

    # Check lambda fits the characteristic equation
    np.testing.assert_allclose(
        x**2 / (a**2 + lambda_) + y**2 / (b**2 + lambda_) + z**2 / (c**2 + lambda_),
        1.0,
    )


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [(3.0, 2.0, 2.0), (1.0, 2.0, 2.0), (3.0, 2.0, 1.0)],
    ids=["prolate", "oblate", "triaxial"],
)
def test_zero_cases_for_lambda(a, b, c):
    """
    Test that lambda calculation produces acceptable values in cases which are
    may throw zero errors.
    """
    x, y, z = 0, 0, 5
    lmbda = calculate_lambda(x, y, z, a, b, c)
    lmbda_eqn = z**2 - c**2
    np.testing.assert_allclose(lmbda, lmbda_eqn)

    x, y, z = 0, 5, 0
    lmbda = calculate_lambda(x, y, z, a, b, c)
    lmbda_eqn = y**2 - b**2
    np.testing.assert_allclose(lmbda, lmbda_eqn)

    x, y, z = 5, 0, 0
    lmbda = calculate_lambda(x, y, z, a, b, c)
    lmbda_eqn = x**2 - a**2
    np.testing.assert_allclose(lmbda, lmbda_eqn)


@pytest.mark.parametrize("zero_coord", ["x", "y", "z"])
@pytest.mark.parametrize(
    ("a", "b", "c"),
    [(3.4, 2.2, 2.2), (1.1, 2.8, 2.8), (3.4, 2.2, 1.1)],
    ids=["prolate", "oblate", "triaxial"],
)
def test_second_order_equations(a, b, c, zero_coord):
    """
    Test lambda calculation against the solutions for its second order
    characteristic equations that take place when only one of the coordinates
    is zero.
    """
    if zero_coord == "z":
        x, y, z = 5.4, 8.1, 0.0
        lambda_ = calculate_lambda(x, y, z, a, b, c)
        p1 = a**2 + b**2 - x**2 - y**2
        p0 = a**2 * b**2 - x**2 * b**2 - y**2 * a**2
    elif zero_coord == "y":
        x, y, z = 5.4, 0.0, 4.5
        lambda_ = calculate_lambda(x, y, z, a, b, c)
        p1 = a**2 + c**2 - x**2 - z**2
        p0 = a**2 * c**2 - x**2 * c**2 - z**2 * a**2
    else:
        x, y, z = 0.0, 8.1, 4.5
        lambda_ = calculate_lambda(x, y, z, a, b, c)
        p1 = b**2 + c**2 - y**2 - z**2
        p0 = b**2 * c**2 - y**2 * c**2 - z**2 * b**2
    expected_lambda = 0.5 * (np.sqrt(p1**2 - 4 * p0) - p1)
    np.testing.assert_allclose(expected_lambda, lambda_)
