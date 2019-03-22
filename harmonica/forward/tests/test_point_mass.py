"""
Test forward modellig for point masses.
"""
import numpy as np
import numpy.testing as npt

from ..point_mass import point_mass_gravity
from harmonica.constants import GRAVITATIONAL_CONST


def test_point_mass_on_equator():
    "Check gravitational fields of point mass"
    point_mass = [0.0, 0.0, 0.0]
    mass = 1.0
    height = np.logspace(1, 5, 5, dtype="float64")
    longitude = np.zeros(height.size)
    latitude = np.zeros(height.size)
    potential = point_mass_gravity(
        [longitude, latitude, height], point_mass, mass, "potential"
    )
    potential_analytical = GRAVITATIONAL_CONST * mass / height
    npt.assert_allclose(potential, potential_analytical)
    gz = point_mass_gravity([longitude, latitude, height], point_mass, mass, "gz")
    gz_analytical = -GRAVITATIONAL_CONST * mass / height ** 2
    # Convert to mGal
    gz_analytical *= 1e5
    npt.assert_allclose(gz, gz_analytical)
