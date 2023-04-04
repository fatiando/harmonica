# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the EquivalentSourcesSph gridder
"""

import numpy as np
import numpy.testing as npt
import pytest
import verde as vd

from .. import EquivalentSourcesSph, point_gravity
from .utils import run_only_with_numba


@pytest.fixture(name="region")
def fixture_region():
    """
    Return a sample region
    """
    return (-70, -60, -40, -30)


@pytest.fixture(name="points")
def fixture_points(region):
    """
    Return the coordinates of some sample point masses
    """
    radius = 6400e3
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=radius - 1e3)
    return points


@pytest.fixture(name="masses")
def fixture_masses(region, points):
    """
    Return the masses some sample point masses
    """
    return vd.synthetic.CheckerBoard(amplitude=1e13, region=region).predict(points)


@pytest.fixture(name="coordinates_small")
def fixture_coordinates_small(region):
    """
    Return a small set of 25 coordinates and variable elevation
    """
    shape = (5, 5)
    longitude, latitude = vd.grid_coordinates(region=region, shape=shape)
    radius = 6400e3 + np.arange(25, dtype=float).reshape(shape)
    coordinates = (longitude, latitude, radius)
    return coordinates


@pytest.fixture(name="data_small")
def fixture_data_small(points, masses, coordinates_small):
    """
    Return some sample data for the small set of coordinates
    """
    return point_gravity(
        coordinates_small, points, masses, field="g_z", coordinate_system="spherical"
    )


@run_only_with_numba
def test_equivalent_sources_spherical():
    """
    Check that predictions are reasonable when interpolating from one grid to
    a denser grid.
    """
    region = (-70, -60, -40, -30)
    radius = 6400e3
    # Build synthetic point masses
    points = vd.grid_coordinates(
        region=region, shape=(6, 6), extra_coords=radius - 500e3
    )
    masses = vd.synthetic.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(
        region=region, shape=(40, 40), extra_coords=radius
    )
    # Get synthetic data
    data = point_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The interpolation should be perfect on the data points
    eqs = EquivalentSourcesSph(relative_depth=500e3)
    eqs.fit(coordinates, data)
    npt.assert_allclose(data, eqs.predict(coordinates), rtol=1.3e-5)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    upward = radius
    shape = (60, 60)
    grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_gravity(
        grid_coords, points, masses, field="g_z", coordinate_system="spherical"
    )
    npt.assert_allclose(true, eqs.predict(grid_coords), rtol=1e-3)

    # Test grid method
    grid = eqs.grid(grid_coords)
    npt.assert_allclose(true, grid.scalars, rtol=1e-3)


def test_equivalent_sources_small_data_spherical():
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
    masses = vd.synthetic.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(8, 8), extra_coords=radius)
    # Get synthetic data
    data = point_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The interpolation should be perfect on the data points
    eqs = EquivalentSourcesSph(relative_depth=500e3)
    eqs.fit(coordinates, data)
    npt.assert_allclose(data, eqs.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    tmp = [i.ravel() for i in coordinates]
    npt.assert_allclose(tmp[:2], eqs.points_[:2], rtol=1e-5)
    npt.assert_allclose(tmp[2] - 500e3, eqs.points_[2], rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    upward = radius + 2e3
    shape = (8, 8)
    grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_gravity(
        grid_coords, points, masses, field="g_z", coordinate_system="spherical"
    )
    npt.assert_allclose(true, eqs.predict(grid_coords), rtol=0.05)

    # Test grid method
    grid = eqs.grid(grid_coords)
    npt.assert_allclose(true, grid.scalars, rtol=0.05)


def test_equivalent_sources_custom_points_spherical():
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
    masses = vd.synthetic.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(5, 5), extra_coords=radius)
    # Get synthetic data
    data = point_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # Pass a custom set of point sources
    points_custom = tuple(
        i.ravel()
        for i in vd.grid_coordinates(
            region=region, shape=(3, 3), extra_coords=radius - 500e3
        )
    )
    eqs = EquivalentSourcesSph(points=points_custom)
    eqs.fit(coordinates, data)

    # Check that the proper source locations were set
    npt.assert_allclose(points_custom, eqs.points_, rtol=1e-5)


def test_equivalent_sources_scatter_not_implemented():
    """
    Check if scatter method raises a NotImplementedError
    """
    eqs = EquivalentSourcesSph()
    with pytest.raises(NotImplementedError):
        eqs.scatter()


def test_equivalent_sources_profile_not_implemented():
    """
    Check if scatter method raises a NotImplementedError
    """
    eqs = EquivalentSourcesSph()
    with pytest.raises(NotImplementedError):
        eqs.profile(point1=(1, 1), point2=(2, 2), size=3)


def test_equivalent_sources_spherical_no_projection():
    """
    Check if projection is not a valid argument of grid method
    """
    eqs = EquivalentSourcesSph()
    with pytest.raises(TypeError):
        eqs.grid(upward=10, projection=lambda a, b: (a * 2, b * 2))


@run_only_with_numba
def test_equivalent_sources_spherical_parallel():
    """
    Check predictions when parallel is enabled and disabled
    """
    region = (-70, -60, -40, -30)
    radius = 6400e3
    # Build synthetic point masses
    points = vd.grid_coordinates(
        region=region, shape=(6, 6), extra_coords=radius - 500e3
    )
    masses = vd.synthetic.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(
        region=region, shape=(40, 40), extra_coords=radius
    )
    # Get synthetic data
    data = point_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The predictions should be equal whether are run in parallel or in serial
    relative_depth = 500e3
    eqs_serial = EquivalentSourcesSph(relative_depth=relative_depth, parallel=False)
    eqs_serial.fit(coordinates, data)
    eqs_parallel = EquivalentSourcesSph(relative_depth=relative_depth, parallel=True)
    eqs_parallel.fit(coordinates, data)

    upward = radius
    shape = (60, 60)
    grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    grid_serial = eqs_serial.grid(grid_coords)
    grid_parallel = eqs_parallel.grid(grid_coords)
    npt.assert_allclose(grid_serial.scalars, grid_parallel.scalars, rtol=1e-7)


@pytest.mark.parametrize(
    "deprecated_args",
    (
        dict(upward=5e3, spacing=1),
        dict(upward=5e3, shape=(6, 6)),
        dict(upward=5e3, spacing=1, region=(-75, -55, -40, -30)),
        dict(upward=5e3, shape=(6, 6), region=(-75, -55, -40, -30)),
    ),
)
def test_error_deprecated_args(coordinates_small, data_small, region, deprecated_args):
    """
    Test if EquivalentSourcesSph.grid raises error on deprecated arguments
    """
    # Define sample equivalent sources and fit against synthetic data
    eqs = EquivalentSourcesSph().fit(coordinates_small, data_small)
    # Build a target grid
    grid_coords = vd.grid_coordinates(region=region, shape=(4, 4), extra_coords=2e3)
    # Try to grid passing deprecated arguments
    msg = "The 'upward', 'region', 'shape' and 'spacing' arguments have been"
    with pytest.raises(ValueError, match=msg):
        eqs.grid(coordinates=grid_coords, **deprecated_args)


def test_error_ignored_args(coordinates_small, data_small, region):
    """
    Test if EquivalentSourcesSph.grid raises warning on ignored arguments
    """
    # Define sample equivalent sources and fit against synthetic data
    eqs = EquivalentSourcesSph().fit(coordinates_small, data_small)
    # Build a target grid
    grid_coords = vd.grid_coordinates(region=region, shape=(4, 4), extra_coords=2e3)
    # Try to grid passing kwarg arguments that will be ignored
    msg = "The 'bla' arguments are being ignored."
    with pytest.warns(FutureWarning, match=msg):
        eqs.grid(coordinates=grid_coords, bla="bla")
