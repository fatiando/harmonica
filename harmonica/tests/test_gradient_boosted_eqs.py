# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions for gradient-boosted equivalent sources
"""
import numpy as np
import numpy.testing as npt
import pytest
import verde as vd

from .. import EquivalentSourcesGB, point_gravity
from ..equivalent_sources.gradient_boosted import _get_region_data_sources
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
    return vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)


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
    shape = (8, 8)
    return vd.grid_coordinates(region=region, shape=shape, extra_coords=0)


@pytest.fixture(name="data_small")
def fixture_data_small(points, masses, coordinates_small):
    """
    Return some sample data for the small set of coordinates
    """
    return point_gravity(coordinates_small, points, masses, field="g_z")


@pytest.mark.parametrize(
    "data_region, sources_region, expected_region",
    (
        [(-1, 1, -1, 1), (-2, 2, -2, 2), (-2, 2, -2, 2)],
        [(-3, 3, -3, 3), (-3, 3, -3, 3), (-3, 3, -3, 3)],
        [(-2, 2, -2, 2), (-2, 1, -2, 1), (-2, 2, -2, 2)],
        [(-2, 2, -2, 2), (-2, 1, -1, 2), (-2, 2, -2, 2)],
        [(-2, 2, -2, 2), (1, 2, 1, 2), (-2, 2, -2, 2)],
        [(-2, 2, -2, 2), (1, 2, -2, 1), (-2, 2, -2, 2)],
    ),
)
def test_get_region_data_sources(data_region, sources_region, expected_region):
    """
    Test the EquivalentSourcesGB._get_region_data_sources method
    """
    shape = (8, 10)
    coordinates = vd.grid_coordinates(data_region, shape=shape)
    points = vd.grid_coordinates(sources_region, shape=shape)
    region = _get_region_data_sources(coordinates, points)
    npt.assert_allclose(region, expected_region)


def test_custom_points(region, coordinates_small, data_small):
    """
    Check that passing in custom points works and actually uses the points
    """
    # Pass a custom set of point sources
    points_custom = tuple(
        i.ravel()
        for i in vd.grid_coordinates(region=region, shape=(3, 3), extra_coords=-550)
    )
    eqs = EquivalentSourcesGB(points=points_custom, window_size=500)
    eqs.fit(coordinates_small, data_small)
    # Check that the proper source locations were set
    npt.assert_allclose(points_custom, eqs.points_, rtol=1e-5)


@pytest.mark.parametrize("spacing", (100, 500, 1e3))
@pytest.mark.parametrize("window_size", (1e3, 2e3, 4e3))
@pytest.mark.parametrize("dtype, itemsize", [("float32", 4), ("float64", 8)])
def test_memory_estimation(spacing, window_size, dtype, itemsize):
    """
    Test the estimate_required_memory class method
    """
    region = (-1e3, 5e3, 2e3, 8e3)
    coordinates = vd.grid_coordinates(region=region, spacing=spacing, extra_coords=0)
    # Compute expected required memory
    sources_p_window = (int(window_size / spacing) + 1) ** 2
    data_p_window = (int(window_size / spacing) + 1) ** 2
    expected_required_memory = data_p_window * sources_p_window * itemsize
    # Estimate required memory
    eqs = EquivalentSourcesGB(window_size=window_size, dtype=dtype)
    required_memory = eqs.estimate_required_memory(coordinates)
    assert required_memory == expected_required_memory


# -----------------------------------------------------------------------
# Test the fitting and predictions of gradient-boosted equivalent sources
# -----------------------------------------------------------------------


@pytest.mark.use_numba
@pytest.mark.parametrize("weights", [None, np.ones((8, 8))])
def test_gb_eqs_small_data(coordinates_small, data_small, weights):
    """
    Check predictions against synthetic data using few data points for speed
    """
    # The interpolation should be good enought on the data points
    # Gradient-boosted equivalent sources don't perform well on small data, so
    # we will check if the error is no larger than 1mGal
    # (sample data ranges from approximately -7mGal to 7mGal)
    eqs = EquivalentSourcesGB(depth=1e3, damping=None, window_size=1e3, random_state=42)
    eqs.fit(coordinates_small, data_small, weights=weights)
    # Error tolerance is 5% of the maximum data.
    npt.assert_allclose(
        data_small,
        eqs.predict(coordinates_small),
        rtol=0,
        atol=0.05 * vd.maxabs(data_small),
    )


@run_only_with_numba
def test_gradient_boosted_eqs_single_window(region, points, masses, coordinates, data):
    """
    Test GB eq-sources with a single window that covers the whole region
    """
    eqs = EquivalentSourcesGB(window_size=region[1] - region[0])
    eqs.fit(coordinates, data)
    npt.assert_allclose(data, eqs.predict(coordinates), rtol=1e-5)
    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region, shape=(60, 60), extra_coords=0)
    true = point_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eqs.predict(grid), rtol=1e-3)


@run_only_with_numba
def test_gradient_boosted_eqs_predictions(region, points, masses, coordinates, data):
    """
    Test GB eq-sources predictions
    """
    # The interpolation should be sufficiently accurate on the data points
    eqs = EquivalentSourcesGB(window_size=1e3, depth=1e3, damping=None, random_state=42)
    eqs.fit(coordinates, data)
    # Error tolerance is 2% of the maximum data.
    npt.assert_allclose(
        data, eqs.predict(coordinates), rtol=0, atol=0.02 * vd.maxabs(data)
    )

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region, shape=(60, 60), extra_coords=0)
    true = point_gravity(grid, points, masses, field="g_z")
    # Error tolerance is 2% of the maximum data.
    npt.assert_allclose(true, eqs.predict(grid), rtol=0, atol=0.02 * vd.maxabs(true))


@run_only_with_numba
def test_gradient_boosted_eqs_random_state(coordinates, data):
    """
    Check if EquivalentSourcesGB produces same result by setting random_state
    """
    # Initialize two EquivalentSourcesGB with the same random_state
    eqs_a = EquivalentSourcesGB(window_size=500, random_state=0)
    eqs_a.fit(coordinates, data)
    eqs_b = EquivalentSourcesGB(window_size=500, random_state=0)
    eqs_b.fit(coordinates, data)

    # Check if fitted coefficients are the same
    npt.assert_allclose(eqs_a.coefs_, eqs_b.coefs_)


def test_same_number_of_windows_data_and_sources():
    """
    Test if _create_windows generates the same num of windows for data and srcs
    """
    spacing = 1
    # Create data points on a large region
    region = (1, 3, 1, 3)
    coordinates = vd.grid_coordinates(region=region, spacing=spacing, extra_coords=0)
    # Create source points on a smaller region
    sources_region = (1.5, 2.5, 1.5, 2.5)
    points = vd.grid_coordinates(
        region=sources_region, spacing=spacing, extra_coords=-10
    )
    # Create EquivalentSourcesGB
    eqs = EquivalentSourcesGB(window_size=spacing)
    # Make EQL believe that it has already created the points
    eqs.points_ = points
    # Create windows for data points and sources
    source_windows, data_windows = eqs._create_windows(coordinates)
    # Check if number of windows are the same
    assert len(source_windows) == len(data_windows)


def test_same_windows_data_and_sources():
    """
    Test if _create_windows generates the same windows for data and sources
    """
    spacing = 1
    # Create data points on a large region
    region = (1, 3, 1, 3)
    coordinates = vd.grid_coordinates(region=region, spacing=spacing, extra_coords=0)
    # Create source points on a subregion
    sources_region = (1, 2, 1, 3)
    points = vd.grid_coordinates(
        region=sources_region, spacing=spacing, extra_coords=-10
    )
    # Create EquivalentSourcesGB
    eqs = EquivalentSourcesGB(window_size=spacing)
    # Make EQL believe that it has already created the points
    eqs.points_ = points
    # Create windows for data points and sources
    # Set suffhle to False so we can compare the windows with expected values
    source_windows, data_windows = eqs._create_windows(coordinates, shuffle=False)
    # Check number of windows
    assert len(source_windows) == 9
    assert len(data_windows) == 9
    # Define expected number of points inside each window for data and sources
    expected_data_windows = np.array([[4, 2, 4], [2, 1, 2], [4, 2, 4]]).ravel()
    expected_source_windows = np.array([[4, 2, 2], [2, 1, 1], [4, 2, 2]]).ravel()
    # Check if the windows were created correctly
    for i, window in enumerate(data_windows):
        assert len(window) == expected_data_windows[i]
    for i, window in enumerate(source_windows):
        assert len(window) == expected_source_windows[i]


# -------------------------------------------------
# Test dtype on gradient_boosted equivalent sources
# -------------------------------------------------


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
    Test dtype argument on EquivalentSources
    """
    # Define the points argument for EquivalentSources
    points = None
    if custom_points:
        points = vd.grid_coordinates(region, spacing=300, extra_coords=-2e3)
    # Define the points argument for EquivalentSources.fit()
    if weights_none:
        weights = None
    # Initialize and fit the equivalent sources
    eqs = EquivalentSourcesGB(
        window_size=1e3,
        damping=damping,
        points=points,
        block_size=block_size,
        dtype=dtype,
    )
    eqs.fit(coordinates, data, weights)
    # Make some predictions
    prediction = eqs.predict(coordinates)
    # Check data type of created objects
    for coord in eqs.points_:
        assert coord.dtype == np.dtype(dtype)
    assert prediction.dtype == np.dtype(dtype)


@run_only_with_numba
def test_gradient_boosted_eqs_float32(coordinates, data):
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid, using float32 as dtype.
    """
    eqs = EquivalentSourcesGB(
        depth=500, damping=None, window_size=1e3, random_state=42, dtype="float32"
    )
    eqs.fit(coordinates, data)
    # Error tolerance is 5% of the maximum data.
    npt.assert_allclose(
        data, eqs.predict(coordinates), rtol=0, atol=0.05 * vd.maxabs(data)
    )
