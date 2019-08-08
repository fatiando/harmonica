"""
Test the EQLHarmonic gridder
"""
import pytest
import numpy as np
import numpy.testing as npt
from verde import scatter_points, grid_coordinates
from verde.datasets.synthetic import CheckerBoard
from verde.base import n_1d_arrays

from .. import EQLHarmonic
from ..equivalent_layer.harmonic import jacobian_numba
from .utils import require_numba


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


@require_numba
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


def test_EQLHarmonic_numba_disabled():
    """
    See if exact solution matches the synthetic data with few sources
    """
    region = (-3e4, -1e4, 5e4, 7e4)
    # Build synthetic point masses
    depth = 1e3
    points = scatter_points(region=region, size=50, random_state=1, extra_coords=depth)
    checker = CheckerBoard(region=region)
    synth_masses = checker.predict(points)
    # Define a random set of observation points
    coordinates = scatter_points(region=region, size=50, random_state=2, extra_coords=0)
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


@pytest.mark.use_numba
def test_jacobian():
    "Test Jacobian matrix under symetric system of point sources"
    easting, northing, upward = grid_coordinates(
        region=[-100, 100, -100, 100], shape=(2, 2), extra_coords=0
    )
    points = n_1d_arrays((easting, northing, upward + 100), n=3)
    coordinates = n_1d_arrays((easting, northing, upward), n=3)
    n_points = points[0].size
    jacobian = np.zeros((n_points, n_points), dtype=points[0].dtype)
    jacobian_numba(coordinates, points, jacobian)
    # All diagonal elements must be equal
    diagonal = np.diag_indices(4)
    npt.assert_allclose(jacobian[diagonal][0], jacobian[diagonal])
    # All anti-diagonal elements must be equal (elements between distant points)
    anti_diagonal = (diagonal[0], diagonal[1][::-1])
    npt.assert_allclose(jacobian[anti_diagonal][0], jacobian[anti_diagonal])
    # All elements corresponding to nearest neigbours must be equal
    nearest_neighbours = np.ones((4, 4), dtype=bool)
    nearest_neighbours[diagonal] = False
    nearest_neighbours[anti_diagonal] = False
    npt.assert_allclose(jacobian[nearest_neighbours][0], jacobian[nearest_neighbours])
