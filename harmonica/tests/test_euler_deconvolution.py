# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy.testing as npt
import verde as vd

from .. import (
    EulerDeconvolution,
    derivative_easting,
    derivative_northing,
    derivative_upward,
    dipole_magnetic,
    magnetic_angles_to_vec,
)
from .._forward.point import point_gravity


def test_euler_with_numeric_derivatives():
    # Add dipole source
    dipole_coordinates = (10e3, 15e3, -10e3)
    dipole_moments = magnetic_angles_to_vec(1.0e14, 0, 0)

    # Add regional field
    inc, dec = -40, 15
    fe, fn, fu = magnetic_angles_to_vec(1, inc, dec)
    region = [-100e3, 100e3, -80e3, 80e3]
    coordinates = vd.grid_coordinates(region, spacing=500, extra_coords=500)
    be, bn, bu = dipole_magnetic(
        coordinates, dipole_coordinates, dipole_moments, field="b"
    )

    # Add a fixed base level
    true_base_level = 200  # nT
    anomaly = (fe * be + fn * bn + fu * bu) + true_base_level

    grid = vd.make_xarray_grid(
        coordinates, anomaly, data_names="tfa", extra_coords_names="upward"
    )
    grid["d_east"] = derivative_easting(grid.tfa)
    grid["d_north"] = derivative_northing(grid.tfa)
    grid["d_up"] = derivative_upward(grid.tfa)
    grid_table = vd.grid_to_table(grid)

    euler = EulerDeconvolution(structural_index=3)

    coordinates = (grid_table.easting, grid_table.northing, grid_table.upward)
    euler.fit(
        (grid_table.easting, grid_table.northing, grid_table.upward),
        (grid_table.tfa, grid_table.d_east, grid_table.d_north, grid_table.d_up),
    )

    npt.assert_allclose(euler.location_, dipole_coordinates, atol=1.0e-3, rtol=1.0e-3)
    npt.assert_allclose(euler.base_level_, true_base_level, atol=1.0e-3, rtol=1.0e-3)


def test_euler_with_analytic_derivatives():
    # Add dipole source
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
