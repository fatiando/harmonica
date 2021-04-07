# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the sample data loading functions.
"""
import os

import numpy.testing as npt

from ..datasets.sample_data import (
    locate,
    fetch_gravity_earth,
    fetch_geoid_earth,
    fetch_topography_earth,
    fetch_britain_magnetic,
    fetch_south_africa_gravity,
    fetch_south_africa_topography,
)


def test_datasets_locate():
    "Make sure the data cache location has the right package name"
    # Fetch a dataset first to make sure that the cache folder exists. Since
    # Pooch 1.1.1 the cache isn't created until a download is requested.
    fetch_gravity_earth()
    path = locate()
    assert os.path.exists(path)
    # This is the most we can check in a platform independent way without
    # testing appdirs itself.
    assert "harmonica" in path


def test_geoid_earth():
    "Sanity checks for the loaded grid"
    grid = fetch_geoid_earth()
    assert grid.geoid.shape == (361, 721)
    npt.assert_allclose(grid.geoid.min(), -106.257344)
    npt.assert_allclose(grid.geoid.max(), 84.722744)
    assert grid.attrs
    assert grid.attrs.get("refsysname") == "WGS84"
    assert grid.attrs.get("max_used_degree") == "1277"
    assert grid.attrs.get("tide_system") == "tide_free"
    assert grid.attrs.get("modelname") == "EIGEN-6C4"


def test_gravity_earth():
    "Sanity checks for the loaded grid"
    grid = fetch_gravity_earth()
    assert grid.gravity.shape == (361, 721)
    npt.assert_allclose(grid.gravity.max(), 9.8018358e05)
    npt.assert_allclose(grid.gravity.min(), 9.7476403e05)
    assert grid.height_over_ell.shape == (361, 721)
    npt.assert_allclose(grid.height_over_ell, 10000)
    assert grid.attrs
    assert grid.attrs.get("refsysname") == "WGS84"
    assert grid.attrs.get("max_used_degree") == "1277"
    assert grid.attrs.get("tide_system") == "tide_free"
    assert grid.attrs.get("modelname") == "EIGEN-6C4"


def test_topography_earth():
    "Sanity checks for the loaded grid"
    grid = fetch_topography_earth()
    assert grid.topography.shape == (361, 721)
    npt.assert_allclose(grid.topography.max(), 5651, atol=1)
    npt.assert_allclose(grid.topography.min(), -8409, atol=1)
    assert grid.attrs
    assert grid.attrs.get("refsysname") == "WGS84"
    assert grid.attrs.get("max_used_degree") == "1277"
    assert grid.attrs.get("modelname") == "etopo1-2250"


def test_britain_magnetic():
    "Sanity checks for the loaded dataset"
    data = fetch_britain_magnetic()
    assert data.shape == (541508, 6)
    npt.assert_allclose(data.longitude.min(), -8.65338)
    npt.assert_allclose(data.longitude.max(), 1.92441)
    npt.assert_allclose(data.latitude.min(), 49.81407)
    npt.assert_allclose(data.latitude.max(), 60.97483)
    npt.assert_allclose(data.total_field_anomaly_nt.min(), -3735)
    npt.assert_allclose(data.total_field_anomaly_nt.max(), 2792)
    npt.assert_allclose(data.altitude_m.min(), 201.0)
    npt.assert_allclose(data.altitude_m.max(), 1545.0)
    assert set(data.survey_area.unique()) == {
        "CA55_NORTH",
        "CA55_SOUTH",
        "CA57",
        "CA58",
        "CA59",
        "CA60",
        "CA63",
        "HG56",
        "HG57",
        "HG58",
        "HG61",
        "HG62",
        "HG64",
        "HG65",
    }


def test_south_africa_gravity():
    "Sanity checks for the loaded dataset"
    data = fetch_south_africa_gravity()
    assert data.shape == (14559, 4)
    npt.assert_allclose(data.longitude.min(), 11.90833)
    npt.assert_allclose(data.longitude.max(), 32.74667)
    npt.assert_allclose(data.latitude.min(), -34.996)
    npt.assert_allclose(data.latitude.max(), -17.33333)
    npt.assert_allclose(data.elevation.min(), -1038.00)
    npt.assert_allclose(data.elevation.max(), 2622.17)
    npt.assert_allclose(data.gravity.min(), 978131.3)
    npt.assert_allclose(data.gravity.max(), 979766.65)


def test_south_africa_topography():
    "Sanity checks for the loaded dataset"
    data = fetch_south_africa_topography()
    assert data.topography.shape == (171, 211)
    npt.assert_allclose(data.topography.min(), -5007.0, atol=0.1)
    npt.assert_allclose(data.topography.max(), 3286.0, atol=0.1)
    npt.assert_allclose(data.longitude.min(), 12)
    npt.assert_allclose(data.longitude.max(), 33)
    npt.assert_allclose(data.latitude.min(), -35)
    npt.assert_allclose(data.latitude.max(), -18)
