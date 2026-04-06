# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import numpy.testing as npt
import pytest
import verde as vd

from .. import (
    EulerDeconvolution,
    EulerInversion,
    derivative_easting,
    derivative_northing,
    derivative_upward,
    dipole_magnetic,
    magnetic_angles_to_vec,
    point_gravity,
    total_field_anomaly,
)


@pytest.mark.parametrize(
    "euler",
    [
        EulerInversion(structural_index=3),
        EulerInversion(),
    ],
    ids=("fixed_SI", "estimate_SI"),
)
def test_euler_inversion_with_numeric_derivatives(euler):
    "Use numerical derivatives for a realistic test of the Euler Inversion."
    # Add dipole source
    dipole_coordinates = (10e3, 15e3, -10e3)
    inc, dec = -40, 15
    dipole_moments = magnetic_angles_to_vec(1.0e14, inc, dec)
    region = [-100e3, 100e3, -80e3, 80e3]
    coordinates = vd.grid_coordinates(region, spacing=500, extra_coords=500)
    b = dipole_magnetic(coordinates, dipole_coordinates, dipole_moments, field="b")
    # Add a fixed base level
    true_base_level = 200  # nT
    anomaly = total_field_anomaly(b, inc, dec) + true_base_level
    anomaly += np.random.default_rng(42).normal(0, 20, anomaly.shape)
    grid = vd.make_xarray_grid(
        coordinates, anomaly, data_names="tfa", extra_coords_names="upward"
    )
    grid["d_east"] = derivative_easting(grid.tfa)
    grid["d_north"] = derivative_northing(grid.tfa)
    grid["d_up"] = derivative_upward(grid.tfa)
    table = vd.grid_to_table(grid)
    # Invert and check the output
    coordinates = (table.easting, table.northing, table.upward)
    euler.fit(
        (table.easting, table.northing, table.upward),
        (table.tfa, table.d_east, table.d_north, table.d_up),
    )
    npt.assert_allclose(euler.location_[:2], dipole_coordinates[:2], rtol=0.01)
    npt.assert_allclose(euler.location_[2], dipole_coordinates[2], rtol=0.05)
    npt.assert_allclose(euler.base_level_, true_base_level, rtol=0.01)
    assert euler.structural_index_ == 3


@pytest.mark.parametrize(
    "euler",
    [
        EulerInversion(structural_index=2),
        EulerInversion(),
    ],
    ids=("fixed_SI", "estimate_SI"),
)
def test_euler_inversion_with_analytic_derivatives(euler):
    "Use analytical derivatives to test Euler Inversion without errors from FFT filters."
    # Add point source
    masses_coordinates = (10e3, 15e3, -7e3)
    masses = 1.0e14
    region = [-100e3, 100e3, -80e3, 80e3]
    coordinates = vd.grid_coordinates(region, spacing=500, extra_coords=500)
    gz = point_gravity(coordinates, masses_coordinates, masses, field="g_z")
    random = np.random.default_rng(42)
    gz += random.normal(0, 0.01 * np.abs(gz).max(), gz.shape)
    base_level = 20
    gz += base_level
    # Convert Eötvös to mGal because derivatives must be in mGal/m
    eotvos2mgal = 1.0e-4
    gzz = (
        -point_gravity(coordinates, masses_coordinates, masses, field="g_zz")
        * eotvos2mgal
    )
    gze = (
        point_gravity(coordinates, masses_coordinates, masses, field="g_ez")
        * eotvos2mgal
    )
    gzn = (
        point_gravity(coordinates, masses_coordinates, masses, field="g_nz")
        * eotvos2mgal
    )
    gzz += random.normal(0, 0.02 * np.abs(gzz).max(), gz.shape)
    gze += random.normal(0, 0.02 * np.abs(gze).max(), gz.shape)
    gzn += random.normal(0, 0.02 * np.abs(gzn).max(), gz.shape)
    euler.fit(
        (coordinates[0], coordinates[1], coordinates[2]),
        (gz, gze, gzn, gzz),
    )
    npt.assert_allclose(euler.location_[:2], masses_coordinates[:2], rtol=0.01)
    npt.assert_allclose(euler.location_[2], masses_coordinates[2], rtol=0.05)
    npt.assert_allclose(euler.base_level_, base_level, rtol=0.01)
    assert euler.structural_index_ == 2



@pytest.mark.parametrize(
    "euler",
    [
        EulerInversion(structural_index=3, max_iterations=1),
        EulerInversion(max_iterations=1),
    ],
    ids=("fixed_SI", "estimate_SI"),
)
def test_euler_inversion_convergence_warning(euler):
    "Check that a warning is raised when the method exits without convergence"
    # Add dipole source
    dipole_coordinates = (10e3, 15e3, -10e3)
    inc, dec = -40, 15
    dipole_moments = magnetic_angles_to_vec(1.0e14, inc, dec)
    region = [-100e3, 100e3, -80e3, 80e3]
    coordinates = vd.grid_coordinates(region, spacing=500, extra_coords=500)
    b = dipole_magnetic(coordinates, dipole_coordinates, dipole_moments, field="b")
    # Add a fixed base level
    true_base_level = 200  # nT
    anomaly = total_field_anomaly(b, inc, dec) + true_base_level
    anomaly += np.random.default_rng(42).normal(0, 20, anomaly.shape)
    grid = vd.make_xarray_grid(
        coordinates, anomaly, data_names="tfa", extra_coords_names="upward"
    )
    grid["d_east"] = derivative_easting(grid.tfa)
    grid["d_north"] = derivative_northing(grid.tfa)
    grid["d_up"] = derivative_upward(grid.tfa)
    table = vd.grid_to_table(grid)
    coordinates = (table.easting, table.northing, table.upward)
    with pytest.warns(UserWarning, match="Euler Inversion exited"):
        euler.fit(
            (table.easting, table.northing, table.upward),
            (table.tfa, table.d_east, table.d_north, table.d_up),
        )



def test_euler_deconvolution_with_numeric_derivatives():
    "Use numerical derivatives for a realistic test of the Euler Deconvolution."
    # Add dipole source
    dipole_coordinates = (10e3, 15e3, -10e3)
    inc, dec = -40, 15
    dipole_moments = magnetic_angles_to_vec(1.0e14, inc, dec)
    region = [-100e3, 100e3, -80e3, 80e3]
    coordinates = vd.grid_coordinates(region, spacing=500, extra_coords=500)
    b = dipole_magnetic(coordinates, dipole_coordinates, dipole_moments, field="b")
    # Add a fixed base level
    true_base_level = 200  # nT
    anomaly = total_field_anomaly(b, inc, dec) + true_base_level
    grid = vd.make_xarray_grid(
        coordinates, anomaly, data_names="tfa", extra_coords_names="upward"
    )
    grid["d_east"] = derivative_easting(grid.tfa)
    grid["d_north"] = derivative_northing(grid.tfa)
    grid["d_up"] = derivative_upward(grid.tfa)
    table = vd.grid_to_table(grid)
    # Invert and check the output
    euler = EulerDeconvolution(structural_index=3)
    coordinates = (table.easting, table.northing, table.upward)
    euler.fit(
        (table.easting, table.northing, table.upward),
        (table.tfa, table.d_east, table.d_north, table.d_up),
    )
    npt.assert_allclose(euler.location_, dipole_coordinates, rtol=0.01)
    npt.assert_allclose(euler.base_level_, true_base_level, rtol=0.01)


def test_euler_deconvolution_with_analytic_derivatives():
    "Use analytical derivatives to test Euler Deconvolution without errors from FFT filters."
    # Add point source
    masses_coordinates = (10e3, 15e3, -10e3)
    masses = 1.0e12
    region = [-100e3, 100e3, -80e3, 80e3]
    coordinates = vd.grid_coordinates(region, spacing=500, extra_coords=500)
    gz = point_gravity(coordinates, masses_coordinates, masses, field="g_z")
    # Convert Eötvös to mGal because derivatives must be in mGal/m
    eotvos2mgal = 1.0e-4
    gzz = (
        -point_gravity(coordinates, masses_coordinates, masses, field="g_zz")
        * eotvos2mgal
    )
    gze = (
        point_gravity(coordinates, masses_coordinates, masses, field="g_ez")
        * eotvos2mgal
    )
    gzn = (
        point_gravity(coordinates, masses_coordinates, masses, field="g_nz")
        * eotvos2mgal
    )
    euler = EulerDeconvolution(structural_index=2)
    euler.fit(
        (coordinates[0], coordinates[1], coordinates[2]),
        (gz, gze, gzn, gzz),
    )
    npt.assert_allclose(euler.location_, masses_coordinates, atol=1.0e-3, rtol=1.0e-3)
    npt.assert_allclose(euler.base_level_, 0.0, atol=1.0e-3, rtol=1.0e-3)
