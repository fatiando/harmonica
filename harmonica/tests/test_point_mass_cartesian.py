"""
Test forward modelling for point masses.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..constants import GRAVITATIONAL_CONST
from ..forward.point_mass_cartesian import point_mass_gravity_cartesian


def test_invalid_field():
    "Check if an invalid gravitational field is passed as argument"
    point_mass = [0.0, 0.0, -1e3]
    mass = 1.0
    coordinates = np.zeros(3)
    with pytest.raises(ValueError):
        point_mass_gravity_cartesian(
            coordinates, point_mass, mass, "this-field-does-not-exist"
        )


def test_potential_symmetry():
    """
    Test if potential field of a point mass has symmetry
    """
    # Define a single point mass
    point_mass = [1.1, 1.2, 1.3]
    masses = [2670]
    # Define a set of computation points at a fixed distance from the point mass
    distance = 3.3
    easting = point_mass[0] * np.ones(6)
    northing = point_mass[1] * np.ones(6)
    vertical = point_mass[2] * np.ones(6)
    easting[0] += distance
    easting[1] -= distance
    northing[2] += distance
    northing[3] -= distance
    vertical[4] += distance
    vertical[5] -= distance
    coordinates = [easting, northing, vertical]
    # Compute potential gravity field on each computation point
    results = point_mass_gravity_cartesian(
        coordinates, point_mass, masses, field="potential"
    )
    npt.assert_allclose(*results)


def test_g_vertical_symmetry():
    """
    Test if g_vertical field of a point mass has symmetry
    """
    # Define a single point mass
    point_mass = [1.1, 1.2, 1.3]
    masses = [2670]
    # Define a pair of computation points above and bellow the point mass
    distance = 3.3
    easting = point_mass[0] * np.ones(2)
    northing = point_mass[1] * np.ones(2)
    vertical = point_mass[2] * np.ones(2)
    vertical[0] += distance
    vertical[1] -= distance
    coordinates = [easting, northing, vertical]
    # Compute g_vertical gravity field on each computation point
    results = point_mass_gravity_cartesian(
        coordinates, point_mass, masses, field="g_vertical"
    )
    npt.assert_allclose(results[0], -results[1])
