# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test forward modelling for tesseroids with variable density
"""
import numpy as np
import numpy.testing as npt
import harmonica
import pytest

from ..forward._tesseroid_variable_density import (
    straight_line,
    maximum_absolute_diff,
    density_minmax,
    _density_based_discretization,
)


# Define the accuracy threshold for tesseroids (0.1%) as a
# relative error (0.001)
ACCURACY_THRESHOLD = 1e-3


@pytest.fixture(name="bottom")
def fixture_bottom():
    bottom = 2e3
    return bottom


@pytest.fixture(name="top")
def fixture_top():
    top = 5e3
    return top


@pytest.fixture(name="quadratic_params")
def fixture_quadratic_params():
    factor = 1e-3
    vertex_radius = 3e3
    vertex_density = 1900.0
    return factor, vertex_radius, vertex_density


@pytest.fixture(name="quadratic_density")
def fixture_quadratic_density(quadratic_params):
    """
    Return a quadratic density function

    .. math::

        f(x) = a (x - h)^2 + k

    Where :math:`a` is the ``factor``, :math:`h` is the ``vertex_radius``,
    :math:`k` is the ``vertex_density`` and :math:`x` is the ``radius``.

    ::

       +------------------------------------------------+
       |                                               *|
       |                                              * |
       |                                             *  |
       |                                           **   |
       |                                          *     |
       |                                        **      |
       |                                       *        |
       |                                     **         |
       |                                   **           |
       |                                 **             |
       |*                              **               |
       | **                         ***                 |
       |   ****                  ***                    |
       |       ******     *******                       |
       |             *****                              |
       +------------------------------------------------+
        |              |                               |
      bottom     vertex_radius                        top

    """
    factor, vertex_radius, vertex_density = quadratic_params

    def density(radius):
        """Quadratic density function"""
        return factor * (radius - vertex_radius) ** 2 + vertex_density

    return density


@pytest.fixture(name="straight_line_analytic")
def fixture_straight_line_analytic(bottom, top, quadratic_density):
    """
    Return the analytic solution for the straigh line of the quadratic density
    """
    density_bottom, density_top = quadratic_density(bottom), quadratic_density(top)
    slope = (density_top - density_bottom) / (top - bottom)

    def line(radius):
        return slope * (radius - bottom) + density_bottom

    return line


@pytest.fixture(name="max_abs_diff_analytic")
def fixture_max_abs_diff_analytic(
    bottom, top, quadratic_params, quadratic_density, straight_line_analytic
):
    r"""
    Return analytic solution for maximum absolute difference between quadratic
    density and the straight line

    .. math:

        x_m = \frac{m}{2a} + h

    """
    factor, vertex_radius, _ = quadratic_params
    density_bottom, density_top = quadratic_density(bottom), quadratic_density(top)
    slope = (density_top - density_bottom) / (top - bottom)
    radius_split = 0.5 * slope / factor + vertex_radius
    max_diff = np.abs(
        quadratic_density(radius_split) - straight_line_analytic(radius_split)
    )
    return radius_split, max_diff


@pytest.fixture(name="quadratic_density_minmax")
def fixture_quadratic_density_minmax(top, quadratic_params, quadratic_density):
    """
    Return the analytic maximum and minimum value of the quadratic density
    between top and bottom
    """
    _, _, vertex_density = quadratic_params
    minimum = vertex_density
    maximum = quadratic_density(top)
    return minimum, maximum


def test_straight_line(bottom, top, quadratic_density, straight_line_analytic):
    """
    Test the straight_line function
    """
    radii = np.linspace(bottom, top, 51)
    npt.assert_allclose(
        straight_line(radii, quadratic_density, bottom, top),
        straight_line_analytic(radii),
    )


def test_max_abs_diff(bottom, top, quadratic_density, max_abs_diff_analytic):
    """
    Test the maximum absolute difference

    Test against the sine density defined in density_sine_portion fixture.
    The solution is analytic.
    """
    radius_split_expected, max_diff_expected = max_abs_diff_analytic
    radius_split, max_diff = maximum_absolute_diff(quadratic_density, bottom, top)
    npt.assert_allclose(radius_split_expected, radius_split)
    npt.assert_allclose(max_diff_expected, max_diff)


def test_density_minmax(bottom, top, quadratic_density, quadratic_density_minmax):
    """
    Test the density_minmax function
    """
    density_min, density_max = density_minmax(quadratic_density, bottom, top)
    expected_min, expected_max = quadratic_density_minmax
    npt.assert_allclose(density_min, expected_min)
    npt.assert_allclose(density_max, expected_max)


def test_single_density_based_discretization(
    bottom, top, quadratic_density, max_abs_diff_analytic
):
    """
    Test the density-based discretization algorithm
    """
    # Define some dummy horizontal coordinates for the tesseroid
    w, e, s, n = -3, 2, -4, 5
    tesseroid = w, e, s, n, bottom, top
    tesseroids = _density_based_discretization(tesseroid, quadratic_density)
    # With the default value of DELTA=0.1, it should produce only 2 tesseroids
    assert len(tesseroids) == 2
    # Check the horizontal coordinates of the tesseroids
    for tess in tesseroids:
        for coord, original_coord in zip(tess[:4], tesseroid):
            npt.assert_allclose(coord, original_coord)

    # Check the radial coordinates
    # ----------------------------
    # Take the analytical solution for the radius_split
    radius_split, _ = max_abs_diff_analytic
    # The first tesseroid in the list should be the one in the top
    npt.assert_allclose(tesseroids[0][-2], radius_split)
    npt.assert_allclose(tesseroids[0][-1], top)
    # The second tesseroid in the list should be the one in the bottom
    npt.assert_allclose(tesseroids[1][-2], bottom)
    npt.assert_allclose(tesseroids[1][-1], radius_split)


def test_density_based_discret_with_delta(
    bottom,
    top,
    quadratic_density,
):
    """
    Test the density-based discretization algorithm against values of DELTA
    """
    # Define some dummy horizontal coordinates for the tesseroid
    w, e, s, n = -3, 2, -4, 5
    tesseroid = w, e, s, n, bottom, top
    # Define a collection of values for the delta ratio
    deltas = [1e-3, 1e-2, 1e-1, 1e0]
    # Define an empty list to count the number of splits for each delta
    splits = []
    for delta in deltas:
        # Override the DELTA_RATIO global variable
        harmonica.forward._tesseroid_variable_density.DELTA_RATIO = delta
        # Count the splits generated by density based discretization
        splits.append(len(_density_based_discretization(tesseroid, quadratic_density)))
    splits = np.array(splits)
    # Check if numbers of splits increases for lower deltas
    assert (splits[1:] < splits[:-1]).all()


def test_density_based_discret_linear_density():
    """
    Test if density-based discretization generates no splits when linear
    density is passed
    """
    w, e, s, n, bottom, top = -3, 2, -4, 5, 30, 50
    tesseroid = [w, e, s, n, bottom, top]

    def linear_density(radius):
        """Define a dummy linear density"""
        return 3 * radius + 2

    tesseroids = _density_based_discretization(tesseroid, linear_density)
    assert len(tesseroids) == 1
    npt.assert_allclose(tesseroids[0], tesseroid)


def test_density_based_discret_constant_density():
    """
    Test if density-based discretization generates no splits when a constant
    density function is passed (this should not be done IRL, pass an array of
    floats as density instead)
    """
    w, e, s, n, bottom, top = -3, 2, -4, 5, 30, 50
    tesseroid = [w, e, s, n, bottom, top]

    def stupid_constant_density(radius):
        """Define a dummy constant density function"""
        return 3

    tesseroids = _density_based_discretization(tesseroid, stupid_constant_density)
    assert len(tesseroids) == 1
    npt.assert_allclose(tesseroids[0], tesseroid)
