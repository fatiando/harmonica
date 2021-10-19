# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# pylint: disable=protected-access
"""
Test the EquivalentSources gridder
"""
import warnings
import pytest
import numpy as np
import numpy.testing as npt
import xarray.testing as xrt
import verde as vd
import verde.base as vdb

from .. import EquivalentSources, EQLHarmonic, point_mass_gravity
from ..equivalent_sources.cartesian import greens_func_cartesian
from ..equivalent_sources.utils import (
    jacobian_numba_serial,
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
def test_equivalent_sources_cartesian():
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid. Use Cartesian coordinates.
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
    eqs = EquivalentSources()
    eqs.fit(coordinates, data)
    npt.assert_allclose(data, eqs.predict(coordinates), rtol=1e-5)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    upward = 0
    shape = (60, 60)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eqs.predict(grid), rtol=1e-3)

    # Test grid method
    grid = eqs.grid(upward, shape=shape, region=region)
    npt.assert_allclose(true, grid.scalars, rtol=1e-3)

    # Test profile method
    point1 = (region[0], region[2])
    point2 = (region[0], region[3])
    profile = eqs.profile(point1, point2, upward, shape[0])
    true = point_mass_gravity(
        (profile.easting, profile.northing, profile.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, profile.scalars, rtol=1e-3)


def test_equivalent_sources_small_data_cartesian():
    """
    Check predictions against synthetic data using few data points for speed
    Use Cartesian coordinates.
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(8, 8), extra_coords=0)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # The interpolation should be perfect on the data points
    eqs = EquivalentSources(depth=500)
    eqs.fit(coordinates, data)
    npt.assert_allclose(data, eqs.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    tmp = [i.ravel() for i in coordinates]
    npt.assert_allclose(tmp[:2], eqs.points_[:2], rtol=1e-5)
    npt.assert_allclose(tmp[2] - 500, eqs.points_[2], rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    upward = 20
    shape = (8, 8)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eqs.predict(grid), rtol=0.08)

    # Test grid method
    grid = eqs.grid(upward, shape=shape, region=region)
    npt.assert_allclose(true, grid.scalars, rtol=0.08)

    # Test profile method
    point1 = (region[0], region[2])
    point2 = (region[0], region[3])
    profile = eqs.profile(point1, point2, upward, 10)
    true = point_mass_gravity(
        (profile.easting, profile.northing, profile.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, profile.scalars, rtol=0.05)


@pytest.fixture(name="coordinates")
def fixture_coordinates():
    """
    Return a set of sample coordinates intended to be used in tests
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Define a set of observation points with variable elevation coordinates
    easting, northing = vd.grid_coordinates(region=region, shape=(8, 8))
    upward = np.arange(64, dtype=float).reshape((8, 8))
    coordinates = (easting, northing, upward)
    return coordinates


@pytest.mark.parametrize(
    "depth_type, upward_expected",
    [
        ("relative", np.arange(64, dtype=float).reshape((8, 8)) - 1.5e3),
        ("constant", -1.5e3 * np.ones((8, 8))),
    ],
    ids=["relative", "constant"],
)
def test_equivalent_sources_build_points(
    coordinates,
    depth_type,
    upward_expected,
):
    """
    Check if build_points method works as expected
    """
    eqs = EquivalentSources(depth=1.5e3, depth_type=depth_type)
    points = eqs._build_points(coordinates)
    expected = (*coordinates[:2], upward_expected)
    npt.assert_allclose(points, expected)


def test_equivalent_sources_build_points_bacwkards(coordinates):
    """
    Check if the old relative_depth argument is well supported

    This test is intended to check if backward compatibility is working
    correctly. The ``relative_depth`` parameter will be deprecated on the next
    major release.
    """
    depth = 4.5e3
    expected_upward = coordinates[2] - depth
    # Check if FutureWarning is raised after passing relative_depth
    with warnings.catch_warnings(record=True) as warn:
        eqs = EquivalentSources(relative_depth=depth)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, FutureWarning)
    # Check if the `depth` and `depth_type` attributes are well fixed
    npt.assert_allclose(eqs.depth, depth)
    assert eqs.depth_type == "relative"
    # Check if location of sources are correct
    points = eqs._build_points(coordinates)
    expected = (*coordinates[:2], expected_upward)
    npt.assert_allclose(points, expected)


def test_equivalent_sources_invalid_depth_type():
    """
    Check if ValueError is raised if invalid depth_type is passed
    """
    with pytest.raises(ValueError):
        EquivalentSources(depth=300, depth_type="blabla")


def test_equivalent_sources_points_depth():
    """
    Check if the points coordinates are properly defined by the fit method
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points with variable elevation coordinates
    easting, northing = vd.grid_coordinates(region=region, shape=(5, 5))
    upward = np.arange(25, dtype=float).reshape((5, 5))
    coordinates = (easting, northing, upward)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # Test with constant depth
    eqs = EquivalentSources(depth=1.3e3, depth_type="constant")
    eqs.fit(coordinates, data)
    expected_points = vdb.n_1d_arrays(
        (easting, northing, -1.3e3 * np.ones_like(easting)), n=3
    )
    npt.assert_allclose(expected_points, eqs.points_)

    # Test with relative depth
    eqs = EquivalentSources(depth=1.3e3, depth_type="relative")
    eqs.fit(coordinates, data)
    expected_points = vdb.n_1d_arrays((easting, northing, upward - 1.3e3), n=3)
    npt.assert_allclose(expected_points, eqs.points_)

    # Test with invalid depth_type
    eqs = EquivalentSources(
        depth=300, depth_type="constant"
    )  # init with valid depth_type
    eqs.depth_type = "blabla"  # change depth_type afterwards
    points = eqs._build_points(
        vd.grid_coordinates(region=(-1, 1, -1, 1), spacing=0.25, extra_coords=1)
    )
    assert points is None


def test_equivalent_sources_custom_points_cartesian():
    """
    Check that passing in custom points works and actually uses the points
    Use Cartesian coordinates.
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(5, 5), extra_coords=0)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # Pass a custom set of point sources
    points_custom = tuple(
        i.ravel()
        for i in vd.grid_coordinates(region=region, shape=(3, 3), extra_coords=-550)
    )
    eqs = EquivalentSources(points=points_custom)
    eqs.fit(coordinates, data)

    # Check that the proper source locations were set
    npt.assert_allclose(points_custom, eqs.points_, rtol=1e-5)


def test_equivalent_sources_scatter_not_implemented():
    """
    Check if scatter method raises a NotImplementedError
    """
    eqs = EquivalentSources()
    with pytest.raises(NotImplementedError):
        eqs.scatter()


@pytest.mark.use_numba
def test_equivalent_sources_jacobian_cartesian():
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
    jacobian_numba_serial(coordinates, points, jacobian, greens_func_cartesian)
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
def test_equivalent_sources_cartesian_parallel():
    """
    Check predictions when parallel is enabled and disabled
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(40, 40), extra_coords=0)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # The predictions should be equal whether are run in parallel or in serial
    eqs_serial = EquivalentSources(parallel=False)
    eqs_serial.fit(coordinates, data)
    eqs_parallel = EquivalentSources(parallel=True)
    eqs_parallel.fit(coordinates, data)

    upward = 0
    shape = (60, 60)
    grid_serial = eqs_serial.grid(upward, shape=shape, region=region)
    grid_parallel = eqs_parallel.grid(upward, shape=shape, region=region)
    npt.assert_allclose(grid_serial.scalars, grid_parallel.scalars, rtol=1e-7)


@pytest.mark.parametrize("depth_type", ("constant", "relative"))
def test_backward_eqlharmonic(depth_type):
    """
    Check backward compatibility with to-be-deprecated EQLHarmonic class

    Check if FutureWarning is raised on initialization
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points with variable elevation coordinates
    easting, northing = vd.grid_coordinates(region=region, shape=(5, 5))
    upward = np.arange(25, dtype=float).reshape((5, 5))
    coordinates = (easting, northing, upward)
    # Get synthetic data
    data = point_mass_gravity(coordinates, points, masses, field="g_z")

    # Fit EquivalentSources instance
    eqs = EquivalentSources(depth=1.3e3, depth_type=depth_type)
    eqs.fit(coordinates, data)

    # Fit deprecated EQLHarmonic instance
    # (check if FutureWarning is raised)
    with warnings.catch_warnings(record=True) as warn:
        eql_harmonic = EQLHarmonic(depth=1.3e3, depth_type=depth_type)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, FutureWarning)
    eql_harmonic.fit(coordinates, data)

    # Check if both gridders are equivalent
    npt.assert_allclose(eqs.points_, eql_harmonic.points_)
    shape = (8, 8)
    xrt.assert_allclose(
        eqs.grid(upward=2e3, shape=shape, region=region),
        eql_harmonic.grid(upward=2e3, shape=shape, region=region),
    )
