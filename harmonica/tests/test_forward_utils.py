# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test utils functions for forward modelling
"""
import pytest
import numpy as np
import numpy.testing as npt
import boule as bl

from ..forward.utils import distance, check_coordinate_system


@pytest.mark.use_numba
def test_distance():
    "Test if computated is distance is right"
    # Cartesian coordinate system
    point_a = (1.1, 1.2, 1.3)
    point_b = (1.1, 1.2, 2.4)
    npt.assert_allclose(distance(point_a, point_b, coordinate_system="cartesian"), 1.1)
    # Spherical coordinate system
    point_a = (32.3, 40.1, 1e4)
    point_b = (32.3, 40.1, 1e4 + 100)
    npt.assert_allclose(distance(point_a, point_b, coordinate_system="spherical"), 100)
    # Geodetic coordinate system
    point_a = (-71.3, 33.5, 1e4)
    point_b = (-71.3, 33.5, 1e4 + 100)
    npt.assert_allclose(
        distance(point_a, point_b, coordinate_system="geodetic", ellipsoid=bl.WGS84),
        100,
        rtol=1e-6,
    )


def test_distance_invalid_coordinate_system():
    "Check if invalid coordinate system is passed to distance function"
    point_a = (0, 0, 0)
    point_b = (1, 1, 1)
    with pytest.raises(ValueError):
        distance(point_a, point_b, "this-is-not-a-valid-coordinate-system")


def test_check_coordinate_system():
    "Check if invalid coordinate system is passed to _check_coordinate_system"
    with pytest.raises(ValueError):
        check_coordinate_system("this-is-not-a-valid-coordinate-system")


def test_geodetic_distance_vs_spherical():
    """
    Compare geodetic distance vs spherical distance after conversion

    Test if the closed-form formula for computing the Euclidean distance
    between two points given in geodetic coordinates is equal to the same
    distance computed by converting the points to spherical coordinates and
    using the ``distance_spherical`` function.
    """
    # Initialize the WGS84 ellipsoid
    ellipsoid = bl.WGS84
    # Define two points in geodetic coordinates
    point_a = (-69.3, -36.4, 405)
    point_b = (-71.2, -33.3, 1025)
    # Compute distance using closed-form formula
    dist = distance(point_a, point_b, coordinate_system="geodetic", ellipsoid=ellipsoid)
    # Convert points to spherical coordinates
    point_a_sph = ellipsoid.geodetic_to_spherical(*point_a)
    point_b_sph = ellipsoid.geodetic_to_spherical(*point_b)
    # Compute distance using these converted points
    dist_sph = distance(point_a_sph, point_b_sph, coordinate_system="spherical")
    npt.assert_allclose(dist, dist_sph)


def test_geodetic_distance_equator_poles():
    """
    Check geodetic distance between points in equator and the poles

    The Euclidean distance between a point on the equator and a point on the
    pole, both located on the surface of the ellipsoid and with the same
    longitude can computed through the semimajor and semiminor axis of the
    ellipsoid.
    """
    # Initialize the WGS84 ellipsoid
    ellipsoid = bl.WGS84
    # Compute the expected distance between the two points
    expected_distance = np.sqrt(
        ellipsoid.semimajor_axis ** 2 + ellipsoid.semiminor_axis ** 2
    )
    # Compute distance for different longitudes and alternate between points on
    # both poles
    for pole_lat in (-90, 90):
        for lon in np.linspace(0, 350, 36):
            point_equator = (lon, 0, 0)
            point_pole = (lon, pole_lat, 0)
            npt.assert_allclose(
                expected_distance,
                distance(
                    point_equator,
                    point_pole,
                    coordinate_system="geodetic",
                    ellipsoid=ellipsoid,
                ),
            )
