# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the EquivalentSources gridder
"""
from collections.abc import Iterable

import numpy as np
import numpy.testing as npt
import pytest
import verde as vd
import verde.base as vdb

from .. import EquivalentSources, point_gravity
from .._equivalent_sources.cartesian import greens_func_cartesian
from .._equivalent_sources.utils import jacobian_numba_serial
from .utils import run_only_with_numba


@pytest.fixture(name="region")
def fixture_region():
    """
    Return a sample region
    """
    return (-3e3, -1e3, 5e3, 7e3)


@pytest.fixture(name="coordinates")
def fixture_coordinates(region):
    """
    Return a set of sample coordinates at zero height
    """
    shape = (40, 40)
    return vd.grid_coordinates(region=region, shape=shape, extra_coords=0)


@pytest.fixture(name="points")
def fixture_points(region):
    """
    Return the coordinates of some sample point masses
    """
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    return points


@pytest.fixture(name="masses")
def fixture_masses(region, points):
    """
    Return the masses some sample point masses
    """
    return vd.synthetic.CheckerBoard(amplitude=1e13, region=region).predict(points)


@pytest.fixture(name="data")
def fixture_data(coordinates, points, masses):
    """
    Return some sample data
    """
    return point_gravity(coordinates, points, masses, field="g_z")


@pytest.fixture(name="weights")
def fixture_weights(data):
    """
    Return some sample data
    """
    return np.ones_like(data)


@pytest.fixture(name="coordinates_small")
def fixture_coordinates_small(region):
    """
    Return a small set of 25 coordinates and variable elevation
    """
    shape = (5, 5)
    easting, northing = vd.grid_coordinates(region=region, shape=shape)
    upward = np.arange(25, dtype=float).reshape(shape)
    coordinates = (easting, northing, upward)
    return coordinates


@pytest.fixture(name="data_small")
def fixture_data_small(points, masses, coordinates_small):
    """
    Return some sample data for the small set of coordinates
    """
    return point_gravity(coordinates_small, points, masses, field="g_z")


@pytest.fixture(name="coordinates_9x9")
def fixture_coordinates_9x9(region):
    """
    Return a small set of 81 coordinates and variable elevation
    """
    shape = (9, 9)
    easting, northing = vd.grid_coordinates(region, shape=shape)
    upward = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
    coordinates = (easting, northing, upward)
    return coordinates


@run_only_with_numba
@pytest.mark.parametrize("dtype", ("default", "float32"))
def test_equivalent_sources_cartesian(region, points, masses, coordinates, data, dtype):
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid. Use Cartesian coordinates.
    """
    # Set absolute tolerances for tests based on dtype (float32 should be less
    # accurate)
    if dtype == "float32":
        kwargs = dict(dtype=dtype)
        atol = 1.7e-3 * vd.maxabs(data)
    else:
        kwargs = {}
        atol = 1e-3 * vd.maxabs(data)

    # Fit the equivalent sources
    eqs = EquivalentSources(**kwargs)
    eqs.fit(coordinates, data)

    # The interpolation should be perfect on the data points
    npt.assert_allclose(data, eqs.predict(coordinates), atol=atol)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    upward = 0
    shape = (60, 60)
    grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_gravity(grid_coords, points, masses, field="g_z")
    npt.assert_allclose(true, eqs.predict(grid_coords), atol=atol)

    # Test grid method
    grid = eqs.grid(grid_coords)
    npt.assert_allclose(true, grid.scalars, atol=atol)

    # Test profile method
    point1 = (region[0], region[2])
    point2 = (region[0], region[3])
    profile = eqs.profile(point1, point2, upward, shape[0])
    true = point_gravity(
        (profile.easting, profile.northing, profile.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, profile.scalars, atol=atol)


def test_equivalent_sources_small_data_cartesian(region, points, masses):
    """
    Check predictions against synthetic data using few data points for speed
    Use Cartesian coordinates.
    """
    # Define a small set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(8, 8), extra_coords=0)
    # Get synthetic data
    data = point_gravity(coordinates, points, masses, field="g_z")

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
    grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_gravity(grid_coords, points, masses, field="g_z")
    npt.assert_allclose(true, eqs.predict(grid_coords), rtol=0.08)

    # Test grid method
    grid = eqs.grid(grid_coords)
    npt.assert_allclose(true, grid.scalars, rtol=0.08)

    # Test profile method
    point1 = (region[0], region[2])
    point2 = (region[0], region[3])
    profile = eqs.profile(point1, point2, upward, 10)
    true = point_gravity(
        (profile.easting, profile.northing, profile.upward), points, masses, field="g_z"
    )
    npt.assert_allclose(true, profile.scalars, rtol=0.05)


def test_equivalent_sources_build_points(coordinates):
    """
    Check if build_points method works as expected

    Test only with block-averaging disabled
    """
    depth = 1.5e3
    eqs = EquivalentSources(depth=depth)
    points = eqs._build_points(coordinates)
    upward_expected = coordinates[-1] - depth
    expected = (*coordinates[:2], upward_expected)
    npt.assert_allclose(points, expected)


@pytest.mark.parametrize(
    "block_size", (750, (750, 1e3)), ids=["block_size as float", "block_size as tuple"]
)
def test_block_averaging_coordinates(coordinates_9x9, block_size):
    """
    Test the _block_averaging_coordinates method
    """
    depth = 1.5e3
    eqs = EquivalentSources(depth=depth, block_size=block_size)
    if isinstance(block_size, Iterable):
        expected = (
            [-2500.0, -1375.0, -2500.0, -1375.0, -2500.0, -1375.0],
            [5250.0, 5250.0, 6000.0, 6000.0, 6750.0, 6750.0],
            [11.0, 15.5, 38.0, 42.5, 65.0, 69.5],
        )
    else:
        expected = (
            [-2750, -2000, -1250, -2750, -2000, -1250, -2750, -2000, -1250],
            [5250, 5250, 5250, 6000, 6000, 6000, 6750, 6750, 6750],
            [10.0, 13.0, 16.0, 37.0, 40.0, 43.0, 64.0, 67.0, 70.0],
        )
    npt.assert_allclose(expected, eqs._block_average_coordinates(coordinates_9x9))


def test_build_points_block_average(coordinates_9x9):
    """
    Test the _build_points method with block-averaging
    """
    depth = 1.5e3
    block_size = 750
    eqs = EquivalentSources(depth=depth, block_size=block_size)
    expected = [
        np.array([-2750, -2000, -1250, -2750, -2000, -1250, -2750, -2000, -1250]),
        np.array([5250, 5250, 5250, 6000, 6000, 6000, 6750, 6750, 6750]),
        np.array([10.0, 13.0, 16.0, 37.0, 40.0, 43.0, 64.0, 67.0, 70.0]),
    ]
    expected[-1] -= depth
    npt.assert_allclose(expected, eqs._build_points(coordinates_9x9))


def test_equivalent_sources_points_depth(coordinates_small, data_small):
    """
    Check if the points coordinates are properly defined by the fit method
    """
    easting, northing, upward = coordinates_small[:]

    eqs = EquivalentSources(depth=1.3e3)
    eqs.fit(coordinates_small, data_small)
    expected_points = vdb.n_1d_arrays((easting, northing, upward - 1.3e3), n=3)
    npt.assert_allclose(expected_points, eqs.points_)


def test_equivalent_sources_custom_points_cartesian(region, coordinates, data):
    """
    Check that passing in custom points works and actually uses the points
    Use Cartesian coordinates.
    """
    # Pass a custom set of point sources
    points_custom = tuple(
        i.ravel() for i in vd.grid_coordinates(region, shape=(3, 3), extra_coords=-550)
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


@run_only_with_numba
def test_equivalent_sources_cartesian_parallel(coordinates, data):
    """
    Check predictions when parallel is enabled and disabled
    """
    # The predictions should be equal whether are run in parallel or in serial
    eqs_serial = EquivalentSources(parallel=False)
    eqs_serial.fit(coordinates, data)
    eqs_parallel = EquivalentSources(parallel=True)
    eqs_parallel.fit(coordinates, data)

    grid_serial = eqs_serial.grid(coordinates)
    grid_parallel = eqs_parallel.grid(coordinates)
    npt.assert_allclose(grid_serial.scalars, grid_parallel.scalars, rtol=1e-7)


@run_only_with_numba
@pytest.mark.parametrize("block_size", (None, 500), ids=["no_blocks", "blocks"])
@pytest.mark.parametrize("custom_points", (False, True), ids=["no_points", "points"])
@pytest.mark.parametrize("weights_none", (False, True), ids=["no_weights", "weights"])
@pytest.mark.parametrize("damping", (None, 0.1), ids=["damping_none", "damping"])
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_dtype(
    region,
    coordinates,
    data,
    weights,
    block_size,
    custom_points,
    weights_none,
    damping,
    dtype,
):
    """
    Test if predictions have the dtype passed as argument.
    """
    # Define the points argument for EquivalentSources
    points = (
        vd.grid_coordinates(region, spacing=300, extra_coords=-2e3)
        if custom_points
        else None
    )
    # Define the weights argument for EquivalentSources.fit()
    weights = weights if not weights_none else None
    # Initialize and fit the equivalent sources
    eqs = EquivalentSources(
        damping=damping, points=points, block_size=block_size, dtype=dtype
    )
    eqs.fit(coordinates, data, weights)
    # Ensure predictions have the expected dtype
    prediction = eqs.predict(coordinates)
    assert prediction.dtype == np.dtype(dtype)
    # Locations of sources should be the same dtype as the coordinates
    for coord in eqs.points_:
        assert coord.dtype == coordinates[0].dtype
    # Sources' coefficients should be the same dtype as the coordinates
    assert eqs.coefs_.dtype == coordinates[0].dtype


@pytest.mark.use_numba
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_jacobian_dtype(region, dtype):
    """
    Test dtype of Jacobian when changing dtype argument on EquivalentSources
    """
    # Build a set of custom coordinates
    coordinates = tuple(
        c.ravel() for c in vd.grid_coordinates(region, shape=(10, 10), extra_coords=0)
    )
    # Create custom set of point sources
    points = tuple(
        p.ravel() for p in vd.grid_coordinates(region, shape=(6, 6), extra_coords=-2e3)
    )
    # Initialize and fit the equivalent sources
    eqs = EquivalentSources(points=points, dtype=dtype)
    # Build jacobian matrix
    jacobian = eqs.jacobian(coordinates, points)
    # Check data type of the Jacobian
    assert jacobian.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "deprecated_args",
    (
        dict(upward=5e3, spacing=1),
        dict(upward=5e3, shape=(6, 6)),
        dict(upward=5e3, spacing=1, region=(-4e3, 0, 5e3, 7e3)),
        dict(upward=5e3, shape=(6, 6), region=(-4e3, 0, 5e3, 7e3)),
    ),
)
def test_error_deprecated_args(coordinates_small, data_small, region, deprecated_args):
    """
    Test if EquivalentSources.grid raises error on deprecated arguments
    """
    # Define sample equivalent sources and fit against synthetic data
    eqs = EquivalentSources().fit(coordinates_small, data_small)
    # Build a target grid
    grid_coords = vd.grid_coordinates(region=region, shape=(4, 4), extra_coords=2e3)
    # Try to grid passing deprecated arguments
    msg = "The 'upward', 'region', 'shape' and 'spacing' arguments have been"
    with pytest.raises(ValueError, match=msg):
        eqs.grid(coordinates=grid_coords, **deprecated_args)


def test_error_ignored_args(coordinates_small, data_small, region):
    """
    Test if EquivalentSources.grid raises warning on ignored arguments
    """
    # Define sample equivalent sources and fit against synthetic data
    eqs = EquivalentSources().fit(coordinates_small, data_small)
    # Build a target grid
    grid_coords = vd.grid_coordinates(region=region, shape=(4, 4), extra_coords=2e3)
    # Try to grid passing kwarg arguments that will be ignored
    msg = "The 'bla' arguments are being ignored."
    with pytest.warns(FutureWarning, match=msg):
        eqs.grid(coordinates=grid_coords, bla="bla")


def test_default_depth(coordinates, data):
    """
    Test if the depth of sources is correctly set by the default strategy
    """
    # Get distance to first neighbour in the grid
    easting, northing = coordinates[:2]
    d_easting = easting[1, 1] - easting[0, 0]
    d_northing = northing[1, 1] - northing[0, 0]
    first_neighbour_distance = min(d_easting, d_northing)
    # Fit the equivalent sources with default `depth`
    eqs = EquivalentSources().fit(coordinates, data)
    npt.assert_allclose(eqs.depth_, first_neighbour_distance * 4.5)


def test_invalid_depth():
    """
    Test if error is raised after passing invalid value for depth.
    """
    invalid_depth = "this is not a valid one"
    msg = f"Found invalid 'depth' value equal to '{invalid_depth}'"
    with pytest.raises(ValueError, match=msg):
        EquivalentSources(depth=invalid_depth)
