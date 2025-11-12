# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import pytest

from .._forward.utils_ellipsoids import calculate_lambda


def test_lambda():
    """
    Test that lambda fits characteristic equation.
    """
    x, y, z = 6, 5, 4
    a, b, c = 3, 2, 1
    lmbda = calculate_lambda(x, y, z, a, b, c)

    # test for lambda is within parameters for an ellipsoid
    assert lmbda > -(c**2)

    # check lambda fits the characteristic equation
    np.testing.assert_allclose(
        x**2 / (a**2 + lmbda) + y**2 / (b**2 + lmbda) + z**2 / (c**2 + lmbda),
        1.0,
    )


def test_zero_cases_for_lambda():
    """
    Test that lambda calculation produces acceptbale values in cases which are
    may throw zero errors.
    """
    a = 3
    b = 2
    c = 1

    #######

    x = 0
    y = 0
    z = 5
    lmbda = calculate_lambda(x, y, z, a, b, c)
    lmbda_eqn = z**2 - c**2
    np.testing.assert_allclose(lmbda, lmbda_eqn)

    x = 0
    y = 5
    z = 0
    lmbda1 = calculate_lambda(x, y, z, a, b, c)
    lmbda_eqn1 = y**2 - b**2
    np.testing.assert_allclose(lmbda1, lmbda_eqn1)

    x = 5
    y = 0
    z = 0
    lmbda2 = calculate_lambda(x, y, z, a, b, c)
    lmbda_eqn2 = x**2 - a**2
    np.testing.assert_allclose(lmbda2, lmbda_eqn2)


@pytest.mark.parametrize("zero_coord", ["x", "y", "z"])
def test_second_order_equations(zero_coord):
    """
    Test lambda calculation against the solutions for its second order
    characteristic equations that take place when only one of the coordinates
    is zero.
    """
    a, b, c = 3.4, 2.1, 1.3
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
