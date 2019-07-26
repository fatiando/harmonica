"""
Test utils functions for forward modelling
"""
import pytest
import numpy.testing as npt

from ..forward.utils import distance


@pytest.mark.use_numba
def test_distance_cartesian():
    "Test if computated is distance is right"
    # Cartesian coordinate system
    point_a = (1.1, 1.2, 1.3)
    point_b = (1.1, 1.2, 2.4)
    npt.assert_allclose(distance(point_a, point_b, coordinate_system="cartesian"), 1.1)
    # Spherical coordinate system
    point_a = (32.3, 40.1, 1e4)
    point_b = (32.3, 40.1, 1e4 + 100)
    npt.assert_allclose(distance(point_a, point_b, coordinate_system="spherical"), 100)
