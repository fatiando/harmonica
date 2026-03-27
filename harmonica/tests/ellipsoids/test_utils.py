# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import itertools
import re

import numpy as np
import pytest

from harmonica import Ellipsoid
from harmonica._forward.ellipsoids.utils import (
    calculate_lambda,
    get_semiaxes_rotation_matrix,
    is_almost_a_sphere,
    get_elliptical_integrals,
    _get_elliptical_integrals_triaxial,
    _get_elliptical_integrals_prolate,
    _get_elliptical_integrals_oblate,
)


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [(6.0, 5.0, 5.0), (5.0, 5.0, 4.0), (6.0, 5.0, 4.0)],
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


def test_lambda_unsorted():
    """
    Test error if semiaxes are not sorted.
    """
    a, b, c = 1.0, 2.0, 3.0
    x, y, z = np.meshgrid(*[np.linspace(-10, 10, 41) for _ in range(3)])
    msg = re.escape("Invalid semiaxes not properly sorted")
    with pytest.raises(ValueError, match=msg):
        calculate_lambda(x, y, z, a, b, c)


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [(3.0, 2.0, 2.0), (2.0, 2.0, 1.0), (3.0, 2.0, 1.0)],
    ids=["prolate", "oblate", "triaxial"],
)
def test_zero_cases_for_lambda(a, b, c):
    """
    Test that lambda calculation produces acceptable values in cases which are
    may throw zero errors.
    """
    x, y, z = 0, 0, 5
    lambda_ = calculate_lambda(x, y, z, a, b, c)
    lambda_eqn = z**2 - c**2
    np.testing.assert_allclose(lambda_, lambda_eqn)

    x, y, z = 0, 5, 0
    lambda_ = calculate_lambda(x, y, z, a, b, c)
    lambda_eqn = y**2 - b**2
    np.testing.assert_allclose(lambda_, lambda_eqn)

    x, y, z = 5, 0, 0
    lambda_ = calculate_lambda(x, y, z, a, b, c)
    lambda_eqn = x**2 - a**2
    np.testing.assert_allclose(lambda_, lambda_eqn)


@pytest.mark.parametrize("zero_coord", ["x", "y", "z"])
@pytest.mark.parametrize(
    ("a", "b", "c"),
    [(3.4, 2.2, 2.2), (2.8, 2.8, 1.1), (3.4, 2.2, 1.1)],
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


class TestSemiaxesRotationMatrix:
    """
    Test the ``get_semiaxes_rotation_matrix`` function.
    """

    @pytest.mark.parametrize(("a", "b", "c"), itertools.permutations((3.0, 2.0, 1.0)))
    def test_sorting_of_semiaxes(self, a, b, c):
        """
        Test if the transposed matrix correctly sorts the semiaxes.

        When we apply the transposed matrix to unsorted semiaxes, we should recover the
        semiaxes in the right order. Although some of them might have a minus sign due
        to the rotation, so we'll compare with absolute values.

        For example:

        .. code::python

            matrix.T @ np.array([1.0, 2.0, 3.0])

        Should return:

        .. code::python

            np.array([3.0, 2.0, -1.0])
        """
        ellipsoid = Ellipsoid(a, b, c)
        matrix = get_semiaxes_rotation_matrix(ellipsoid)
        semiaxes = np.array([a, b, c])
        expected = sorted((a, b, c), reverse=True)
        np.testing.assert_allclose(np.abs(matrix.T @ semiaxes), expected)

    def test_example(self):
        a, b, c = 1, 2, 3
        ellipsoid = Ellipsoid(a, b, c)
        matrix = get_semiaxes_rotation_matrix(ellipsoid)
        semiaxes = np.array([a, b, c])
        expected = [c, b, -a]
        np.testing.assert_allclose(matrix.T @ semiaxes, expected)


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        (1, 1, 1, True),  # exact sphere
        (1.00001, 1, 1, True),  # prolate as sphere
        (1.00001, 1.00001, 1, True),  # oblate as sphere
        (3.00002, 3.00001, 3, True),  # triaxial as sphere
        (2, 1, 1, False),  # non-spherical prolate
        (2, 2, 1, False),  # non-spherical oblate
        (3, 2, 1, False),  # non-spherical triaxial
    ],
    ids=[
        "exact sphere",
        "prolate as sphere",
        "oblate as sphere",
        "triaxial as sphere",
        "non-spherical prolate",
        "non-spherical oblate",
        "non-spherical triaxial",
    ],
)
def test_is_almost_a_sphere(a, b, c, expected):
    """Test the ``is_almost_a_sphere`` function."""
    assert is_almost_a_sphere(a, b, c) == expected


class TestExtraErrors:
    """
    Test extra error messages in private functions.

    Some of these errors are never going to be reached by calling the public functions,
    but we keep them as fail-safes. Adding tests for them here to get coverage on them.
    """

    @pytest.mark.parametrize(
        "semiaxes",
        [
            (1.0, 1.0, 1.0),
            (1.00003, 1.00002, 1.00001),
        ],
        ids=["sphere", "almost_sphere"],
    )
    def test_elliptical_integrals_error_on_sphere(self, semiaxes):
        a, b, c = semiaxes
        msg = re.escape("Invalid semiaxes that create (almost) a spherical ellipsoid")
        with pytest.raises(ValueError, match=msg):
            get_elliptical_integrals(a, b, c, lambda_=2.0)  # random lambda

    def test_elliptical_integrals_triaxial(self):
        a, b, c = 2.0, 2.0, 1.0
        msg = re.escape("Invalid semiaxes length (not a > b > c)")
        with pytest.raises(ValueError, match=msg):
            _get_elliptical_integrals_triaxial(a, b, c, lambda_=2.0)  # random lambda

    def test_elliptical_integrals_prolate(self):
        a, b = 1.0, 2.0
        msg = re.escape("Invalid semiaxes length (not a > b)")
        with pytest.raises(ValueError, match=msg):
            _get_elliptical_integrals_prolate(a, b, lambda_=2.0)  # random lambda

    def test_elliptical_integrals_oblate(self):
        b, c = 1.0, 2.0
        msg = re.escape("Invalid semiaxes length (not b > c)")
        with pytest.raises(ValueError, match=msg):
            _get_elliptical_integrals_oblate(b, c, lambda_=2.0)  # random lambda
