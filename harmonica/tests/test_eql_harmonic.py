"""
Test the EQLHarmonic gridder
"""
import numpy as np
import numpy.testing as npt
from verde import scatter_points, grid_coordinates
from verde.datasets.synthetic import CheckerBoard

from .. import EQLHarmonic


def point_mass_gravity_simple(coordinates, points, masses):
    """
    Function to patch the missing harmonica.point_mass_gravity function

    This should be replaced by hm.point_mass_gravity when its PR is merged.
    """
    from ..equivalent_layer.harmonic import predict_numba
    from ..constants import GRAVITATIONAL_CONST

    coeffs = GRAVITATIONAL_CONST * masses
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=coordinates[0].dtype)
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    predict_numba(coordinates, points, coeffs, result)
    return result.reshape(cast.shape)


def test_EQLHarmonic():
    """
    See if exact solution matches the synthetic data
    """
    region = (-3e4, -1e4, 5e4, 7e4)
    # Build synthetic point masses
    depth = 1e3
    points = scatter_points(
        region=region, size=1500, random_state=1, extra_coords=depth
    )
    checker = CheckerBoard(region=region)
    synth_masses = checker.predict(points)
    # Define a random set of observation points
    coordinates = scatter_points(
        region=region, size=1500, random_state=2, extra_coords=0
    )
    # Get synthetic data
    data = point_mass_gravity_simple(coordinates, points, synth_masses)

    # The interpolation should be perfect on the data points
    gridder = EQLHarmonic()
    gridder.fit(coordinates, data)
    npt.assert_allclose(data, gridder.predict(coordinates), rtol=1e-5)

    # Configure new gridder passing the syntetic points
    gridder = EQLHarmonic(points=points)
    gridder.fit(coordinates, data)
    npt.assert_allclose(data, gridder.predict(coordinates), rtol=1e-5)
