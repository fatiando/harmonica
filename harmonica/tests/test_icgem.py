# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Testing ICGEM gdf files loading.
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises, warns

from .. import load_icgem_gdf

MODULE_DIR = Path(__file__).parent
TEST_DATA_DIR = MODULE_DIR / "data"


def test_load_icgem_gdf():
    """Check if load_icgem_gdf reads an ICGEM file with sample data correctly."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    icgem_grd = load_icgem_gdf(fname)

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    shape = (nlat, nlon)
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    true_data = np.array([np.arange(nlon)] * nlat, dtype="float64")
    height = 1100 * np.ones(shape)

    assert icgem_grd.sizes["latitude"] == nlat
    assert icgem_grd.sizes["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.height_over_ell.values)


def test_load_icgem_gdf_open_file():
    """Check if load_icgem_gdf works if given an open file instead of string."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    with fname.open() as open_file:
        icgem_grd = load_icgem_gdf(open_file)

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    shape = (nlat, nlon)
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    true_data = np.array([np.arange(nlon)] * nlat, dtype="float64")
    height = 1100 * np.ones(shape)

    assert icgem_grd.sizes["latitude"] == nlat
    assert icgem_grd.sizes["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.height_over_ell.values)


def test_load_icgem_gdf_with_height():
    """Check if load_icgem_gdf reads an ICGEM file with height column."""
    fname = TEST_DATA_DIR / "icgem-sample-with-height.gdf"
    icgem_grd = load_icgem_gdf(fname)

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    true_data = np.array([np.arange(nlon)] * nlat, dtype="float64")
    glon, glat = np.meshgrid(lon, lat)
    height = glon + glat

    assert icgem_grd.sizes["latitude"] == nlat
    assert icgem_grd.sizes["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.h_over_geoid.values)


def test_load_icgem_gdf_usecols():
    """Check if load_icgem_gdf loads ICGEM file reading only first two columns."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    icgem_grd = load_icgem_gdf(fname, usecols=[0, 1])

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    shape = (nlat, nlon)
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    height = 1100 * np.ones(shape)

    assert icgem_grd.sizes["latitude"] == nlat
    assert icgem_grd.sizes["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(height, icgem_grd.height_over_ell.values)
    assert len(icgem_grd.data_vars) == 1


def test_missing_shape(tmp_path):
    """ICGEM file with missing shape."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    attributes = ["latitude_parallels", "longitude_parallels"]
    for attribute in attributes:
        corrupt = tmp_path / f"missing_shape_{attribute}.gdf"
        with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
            for line in gdf_file:
                if attribute in line:
                    continue
                corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_missing_size(tmp_path):
    """ICGEM file with missing size."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    corrupt = tmp_path / "missing_size.gdf"
    attribute = "number_of_gridpoints"
    with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
        for line in gdf_file:
            if attribute in line:
                continue
            corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_corrupt_shape(tmp_path):
    """ICGEM file with corrupt shape."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    attributes = ["latitude_parallels", "longitude_parallels"]
    for attribute in attributes:
        corrupt = tmp_path / f"missing_shape_{attribute}.gdf"
        with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
            for line in gdf_file:
                if attribute in line:
                    new_value = int(line.split()[1]) + 1
                    new_line = attribute + "\t" + str(new_value) + "\n"
                    corrupt_gdf.write(new_line)
                else:
                    corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_missing_cols_names(tmp_path):
    """ICGEM file with missing cols names."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    corrupt = tmp_path / "missing_cols_names.gdf"
    with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
        for line in gdf_file:
            if "latitude" in line and "longitude" in line:
                continue
            corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_units(tmp_path):
    """ICGEM file with missing units."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    corrupt = tmp_path / "missing_units.gdf"
    with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
        for line in gdf_file:
            if "[mgal]" in line:
                continue
            corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_empty_line(tmp_path):
    """ICGEM file with missing empty line."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    corrupt = tmp_path / "missing_empty_line.gdf"
    with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
        for line in gdf_file:
            if not line.strip():
                continue
            corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_attribute(tmp_path):
    """ICGEM file with one missing attribute (not missing unit)."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    corrupt = tmp_path / "missing_attribute.gdf"
    with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
        for line in gdf_file:
            if "longitude" in line and "latitude" in line:
                parts = line.strip().split()
                corrupt_gdf.write("\t".join(parts[:2]) + "\n")
            else:
                corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_lat_lon_attributes(tmp_path):
    """ICGEM file with missing longitude or latitude attribute."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    attributes = ["longitude", "latitude"]
    for attribute in attributes:
        corrupt = tmp_path / f"missing_{attribute}_attribute.gdf"
        with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
            for line in gdf_file:
                if "longitude" in line and "latitude" in line:
                    new_line = line.replace(attribute, "corrupt")
                    corrupt_gdf.write(new_line)
                else:
                    corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_diff_attrs_vs_cols(tmp_path):
    """ICGEM file with different number of cols vs number of attributes."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    corrupt = tmp_path / "diff_attributes_vs_cols.gdf"
    with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
        for line in gdf_file:
            if ("longitude" in line and "latitude" in line) or "[mgal]" in line:
                parts = line.strip().split()
                corrupt_gdf.write("\t".join(parts[:2]) + "\n")
            else:
                corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_area(tmp_path):
    """ICGEM file with missing area coordinates."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    attributes = [
        "latlimit_north",
        "latlimit_south",
        "longlimit_west",
        "longlimit_east",
    ]
    for attribute in attributes:
        corrupt = tmp_path / f"missing_{attribute}.gdf"
        with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
            for line in gdf_file:
                if attribute in line:
                    continue
                corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_corrupt_area(tmp_path):
    """ICGEM file with area in header mismatch area from data."""
    fname = TEST_DATA_DIR / "icgem-sample.gdf"
    attributes = [
        "latlimit_north",
        "latlimit_south",
        "longlimit_west",
        "longlimit_east",
    ]
    for attribute in attributes:
        corrupt = tmp_path / f"corrupt_{attribute}.gdf"
        with fname.open() as gdf_file, corrupt.open("w") as corrupt_gdf:
            for line in gdf_file:
                if attribute in line:
                    parts = line.split()
                    new_bound = float(parts[1]) + 1.0
                    newline = parts[0] + "\t" + str(new_bound)
                    corrupt_gdf.write(newline)
                else:
                    corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


@pytest.fixture(name="empty_fname")
def fixture_empty_fname(tmp_path):
    """
    Return the path to a temporary empty file.
    """
    empty_fname = tmp_path / "empty.gdf"
    with empty_fname.open("w") as gdf_file:
        gdf_file.write("")
    return empty_fname


def test_empty_file(empty_fname):
    """Empty ICGEM file."""
    error = raises(IOError, match=r"Couldn't read \w+ field from gdf file header")
    warn = warns(UserWarning, match=r"loadtxt")
    with error, warn:
        load_icgem_gdf(empty_fname)
