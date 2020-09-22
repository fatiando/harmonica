"""
Test the EQLHarmonic gridder
"""
import warnings
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd
import verde.base as vdb

from .. import EQLHarmonic, EQLHarmonicSpherical, point_mass_gravity
from ..equivalent_layer.harmonic import (
    jacobian_numba,
    greens_func_cartesian,
    pop_extra_coords,
)
from .utils import require_numba


def test_pop_extra_coords():
    """
    Test _pop_extra_coords private function
    """
    # Check if extra_coords is removed from kwargs
    kwargs = {"bla": 1, "blabla": 2, "extra_coords": 1400.0}
    with warnings.catch_warnings(record=True) as warn:
        pop_extra_coords(kwargs)
        assert len(warn) == 1
        assert issubclass(warn[0].category, UserWarning)
    assert "extra_coords" not in kwargs

    # Check if kwargs is not touched if no extra_coords are present
    kwargs = {"bla": 1, "blabla": 2}
    pop_extra_coords(kwargs)
    assert kwargs == {"bla": 1, "blabla": 2}


@require_numba
def test_eql_harmonic_cartesian():
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid. Use Cartesian coordiantes.
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
    upward = 0
    shape = (60, 60)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=1e-3)

    # Test grid method
    grid = eql.grid(upward, shape=shape, region=region)
    npt.assert_allclose(true, grid.scalars, rtol=1e-3)

    # Test scatter method
    scatter = eql.scatter(upward, size=shape[0], random_state=0)
    true = point_mass_gravity(
        (scatter.easting, scatter.northing, scatter.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, scatter.scalars, rtol=1e-3)

    # Test profile method
    point1 = (region[0], region[2])
    point2 = (region[0], region[3])
    profile = eql.profile(point1, point2, upward, shape[0])
    true = point_mass_gravity(
        (profile.easting, profile.northing, profile.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, profile.scalars, rtol=1e-3)


def test_eql_harmonic_small_data_cartesian():
    """
    Check predictions against synthetic data using few data points for speed
    Use Cartesian coordinates.
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
    upward = 20
    shape = (20, 20)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)

    # Test grid method
    grid = eql.grid(upward, shape=shape, region=region)
    npt.assert_allclose(true, grid.scalars, rtol=0.05)

    # Test scatter method
    scatter = eql.scatter(upward, size=20, random_state=0)
    true = point_mass_gravity(
        (scatter.easting, scatter.northing, scatter.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, scatter.scalars, rtol=0.05)

    # Test profile method
    point1 = (region[0], region[2])
    point2 = (region[0], region[3])
    profile = eql.profile(point1, point2, upward, 10)
    true = point_mass_gravity(
        (profile.easting, profile.northing, profile.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, profile.scalars, rtol=0.05)


def test_eql_harmonic_custom_points_cartesian():
    """
    Check that passing in custom points works and actually uses the points
    Use Cartesian coordinates.
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
    eql = EQLHarmonic(points=src_points)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    npt.assert_allclose(src_points, eql.points_, rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=20)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)


@pytest.mark.use_numba
def test_eql_harmonic_jacobian_cartesian():
    """
    Test Jacobian matrix under symmetric system of point sources.
    Use Cartesian coordinates.
    """
    easting, northing, upward = vd.grid_coordinates(
        region=[-100, 100, -100, 100], shape=(2, 2), extra_coords=0
    )
    points = vdb.n_1d_arrays((easting, northing, upward + 100), n=3)
    coordinates = vdb.n_1d_arrays((easting, northing, upward), n=3)
    n_points = points[0].size
    jacobian = np.zeros((n_points, n_points), dtype=points[0].dtype)
    jacobian_numba(coordinates, points, jacobian, greens_func_cartesian)
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


@require_numba
def test_eql_harmonic_spherical():
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid. Use spherical coordiantes.
    """
    region = (-70, -60, -40, -30)
    radius = 6400e3
    # Build synthetic point masses
    points = vd.grid_coordinates(
        region=region, shape=(6, 6), extra_coords=radius - 500e3
    )
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(
        region=region, shape=(40, 40), extra_coords=radius
    )
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The interpolation should be perfect on the data points
    eql = EQLHarmonicSpherical(relative_depth=500e3)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    upward = radius
    shape = (60, 60)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(
        grid, points, masses, field="g_z", coordinate_system="spherical"
    )
    npt.assert_allclose(true, eql.predict(grid), rtol=1e-3)

    # Test grid method
    grid = eql.grid(upward, shape=shape, region=region)
    npt.assert_allclose(true, grid.scalars, rtol=1e-3)

    # Test scatter method
    scatter = eql.scatter(upward, size=shape[0], random_state=0)
    true = point_mass_gravity(
        (scatter.longitude, scatter.spherical_latitude, scatter.radius),
        points,
        masses,
        field="g_z",
        coordinate_system="spherical",
    )
    npt.assert_allclose(true, scatter.scalars, rtol=1e-3)


def test_eql_harmonic_small_data_spherical():
    """
    Check predictions against synthetic data using few data points for speed
    Use spherical coordinates.
    """
    region = (-70, -60, -40, -30)
    radius = 6400e3
    # Build synthetic point masses
    points = vd.grid_coordinates(
        region=region, shape=(6, 6), extra_coords=radius - 500e3
    )
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(
        region=region, shape=(20, 20), extra_coords=radius
    )
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The interpolation should be perfect on the data points
    eql = EQLHarmonicSpherical(relative_depth=500e3)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    tmp = [i.ravel() for i in coordinates]
    npt.assert_allclose(tmp[:2], eql.points_[:2], rtol=1e-5)
    npt.assert_allclose(tmp[2] - 500e3, eql.points_[2], rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    upward = radius + 2e3
    shape = (20, 20)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(
        grid, points, masses, field="g_z", coordinate_system="spherical"
    )
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)

    # Test grid method
    grid = eql.grid(upward, shape=shape, region=region)
    npt.assert_allclose(true, grid.scalars, rtol=0.05)

    # Test scatter method
    scatter = eql.scatter(upward, size=shape[0], random_state=0)
    true = point_mass_gravity(
        (scatter.longitude, scatter.spherical_latitude, scatter.radius),
        points,
        masses,
        field="g_z",
        coordinate_system="spherical",
    )
    npt.assert_allclose(true, scatter.scalars, rtol=0.05)


def test_eql_harmonic_custom_points_spherical():
    """
    Check that passing in custom points works and actually uses the points
    Use spherical coordinates.
    """
    region = (-70, -60, -40, -30)
    radius = 6400e3
    # Build synthetic point masses
    points = vd.grid_coordinates(
        region=region, shape=(6, 6), extra_coords=radius - 500e3
    )
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(
        region=region, shape=(20, 20), extra_coords=radius
    )
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The interpolation should be perfect on the data points
    src_points = tuple(
        i.ravel()
        for i in vd.grid_coordinates(
            region=region, shape=(20, 20), extra_coords=radius - 500e3
        )
    )
    eql = EQLHarmonicSpherical(points=src_points)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    npt.assert_allclose(src_points, eql.points_, rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=radius + 2e3)
    true = point_mass_gravity(
        grid, points, masses, field="g_z", coordinate_system="spherical"
    )
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)
