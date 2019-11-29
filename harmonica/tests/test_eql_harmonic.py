"""
Test the EQLHarmonic gridder
"""
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd
import verde.base as vdb

from .. import EQLHarmonic, point_mass_gravity
from ..equivalent_layer.harmonic import jacobian_numba
from .utils import require_numba


@require_numba
def test_eql_harmonic():
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid.
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(40, 40), extra_coords=0)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # The interpolation should be perfect on the data points
    eql = EQLHarmonic()
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(60, 60), extra_coords=0)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=1e-3)


def test_eql_harmonic_small_data():
    """
    Check predictions against synthetic data using few data points for speed
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=0)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # The interpolation should be perfect on the data points
    eql = EQLHarmonic(relative_depth=500)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    tmp = [i.ravel() for i in coordinates]
    npt.assert_allclose(tmp[:2], eql.points_[:2], rtol=1e-5)
    npt.assert_allclose(tmp[2] - 500, eql.points_[2], rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=20)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)


def test_eql_harmonic_custom_points():
    """
    Check that passing in custom points works and actually uses the points
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=0)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # The interpolation should be perfect on the data points
    src_points = tuple(
        i.ravel()
        for i in vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=-550)
    )
    eql = EQLHarmonic(relative_depth=500, points=src_points)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    npt.assert_allclose(src_points[:2], eql.points_[:2], rtol=1e-5)
    npt.assert_allclose(-550, eql.points_[2], rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=20)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)


@pytest.mark.use_numba
def test_eql_harmonic_jacobian():
    "Test Jacobian matrix under symmetric system of point sources"
    easting, northing, upward = vd.grid_coordinates(
        region=[-100, 100, -100, 100], shape=(2, 2), extra_coords=0
    )
    points = vdb.n_1d_arrays((easting, northing, upward + 100), n=3)
    coordinates = vdb.n_1d_arrays((easting, northing, upward), n=3)
    n_points = points[0].size
    jacobian = np.zeros((n_points, n_points), dtype=points[0].dtype)
    jacobian_numba(coordinates, points, jacobian)
    # All diagonal elements must be equal
    diagonal = np.diag_indices(4)
    npt.assert_allclose(jacobian[diagonal][0], jacobian[diagonal])
    # All anti-diagonal elements must be equal (elements between distant
    # points)
    anti_diagonal = (diagonal[0], diagonal[1][::-1])
    npt.assert_allclose(jacobian[anti_diagonal][0], jacobian[anti_diagonal])
    # All elements corresponding to nearest neighbors must be equal
    nearest_neighbours = np.ones((4, 4), dtype=bool)
    nearest_neighbours[diagonal] = False
    nearest_neighbours[anti_diagonal] = False
    npt.assert_allclose(jacobian[nearest_neighbours][0], jacobian[nearest_neighbours])
