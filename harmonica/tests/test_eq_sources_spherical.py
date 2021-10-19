# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# pylint: disable=protected-access
"""
Test the EquivalentSourcesSph gridder
"""
import warnings
import pytest
import numpy.testing as npt
import xarray.testing as xrt
import verde as vd

from .. import EquivalentSourcesSph, EQLHarmonicSpherical, point_mass_gravity
from .utils import require_numba


@require_numba
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
    eql = EquivalentSourcesSph(relative_depth=500e3)
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
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(8, 8), extra_coords=radius)
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The interpolation should be perfect on the data points
    eql = EquivalentSourcesSph(relative_depth=500e3)
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Check that the proper source locations were set
    tmp = [i.ravel() for i in coordinates]
    npt.assert_allclose(tmp[:2], eql.points_[:2], rtol=1e-5)
    npt.assert_allclose(tmp[2] - 500e3, eql.points_[2], rtol=1e-5)

    # Gridding at higher altitude should be reasonably accurate when compared
    # to synthetic values
    upward = radius + 2e3
    shape = (8, 8)
    grid = vd.grid_coordinates(region=region, shape=shape, extra_coords=upward)
    true = point_mass_gravity(
        grid, points, masses, field="g_z", coordinate_system="spherical"
    )
    npt.assert_allclose(true, eql.predict(grid), rtol=0.05)

    # Test grid method
    grid = eql.grid(upward, shape=shape, region=region)
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
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(5, 5), extra_coords=radius)
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # Pass a custom set of point sources
    points_custom = tuple(
        i.ravel()
        for i in vd.grid_coordinates(
            region=region, shape=(3, 3), extra_coords=radius - 500e3
        )
    )
    eql = EquivalentSourcesSph(points=points_custom)
    eql.fit(coordinates, data)

    # Check that the proper source locations were set
    npt.assert_allclose(points_custom, eql.points_, rtol=1e-5)


def test_equivalent_sources_scatter_not_implemented():
    """
    Check if scatter method raises a NotImplementedError
    """
    eql = EquivalentSourcesSph()
    with pytest.raises(NotImplementedError):
        eql.scatter()


def test_equivalent_sources_profile_not_implemented():
    """
    Check if scatter method raises a NotImplementedError
    """
    eql = EquivalentSourcesSph()
    with pytest.raises(NotImplementedError):
        eql.profile(point1=(1, 1), point2=(2, 2), size=3)


def test_equivalent_sources_spherical_no_projection():
    """
    Check if projection is not a valid argument of grid method
    """
    eql = EquivalentSourcesSph()
    with pytest.raises(TypeError):
        eql.grid(upward=10, projection=lambda a, b: (a * 2, b * 2))


@require_numba
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
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(
        region=region, shape=(40, 40), extra_coords=radius
    )
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # The predictions should be equal whether are run in parallel or in serial
    relative_depth = 500e3
    eql_serial = EquivalentSourcesSph(relative_depth=relative_depth, parallel=False)
    eql_serial.fit(coordinates, data)
    eql_parallel = EquivalentSourcesSph(relative_depth=relative_depth, parallel=True)
    eql_parallel.fit(coordinates, data)

    upward = radius
    shape = (60, 60)
    grid_serial = eql_serial.grid(upward, shape=shape, region=region)
    grid_parallel = eql_parallel.grid(upward, shape=shape, region=region)
    npt.assert_allclose(grid_serial.scalars, grid_parallel.scalars, rtol=1e-7)


def test_backward_eqlharmonicspherical():
    """
    Check backward compatibility with to-be-deprecated EQLHarmonicSph

    Check if FutureWarning is raised on initialization
    """
    region = (-70, -60, -40, -30)
    radius = 6400e3
    # Build synthetic point masses
    points = vd.grid_coordinates(
        region=region, shape=(6, 6), extra_coords=radius - 500e3
    )
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(8, 8), extra_coords=radius)
    # Get synthetic data
    data = point_mass_gravity(
        coordinates, points, masses, field="g_z", coordinate_system="spherical"
    )

    # Fit EquivalentSourcesSph instance
    eql = EquivalentSourcesSph(relative_depth=1.3e3)
    eql.fit(coordinates, data)

    # Fit deprecated EQLHarmonicSpherical instance
    # (check if FutureWarning is raised)
    with warnings.catch_warnings(record=True) as warn:
        eql_harmonic = EQLHarmonicSpherical(relative_depth=1.3e3)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, FutureWarning)
    eql_harmonic.fit(coordinates, data)

    # Check if both gridders are equivalent
    npt.assert_allclose(eql.points_, eql_harmonic.points_)
    shape = (8, 8)
    xrt.assert_allclose(
        eql.grid(upward=6405e3, shape=shape, region=region),
        eql_harmonic.grid(upward=6405e3, shape=shape, region=region),
    )
