"""
Testing ICGEM gdf files loading.
"""
import os
import numpy as np
import numpy.testing as npt
from pytest import raises

from .. import load_icgem_gdf

MODULE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MODULE_DIR, "data")


def test_load_icgem_gdf():
    "Check if load_icgem_gdf reads an ICGEM file with sample data correctly"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    icgem_grd = load_icgem_gdf(fname)

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    shape = (nlat, nlon)
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    true_data = np.array([np.arange(nlon)] * nlat, dtype="float64")
    height = 1100 * np.ones(shape)

    assert icgem_grd.dims["latitude"] == nlat
    assert icgem_grd.dims["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.height_over_ell.values)


def test_load_icgem_gdf_with_height():
    "Check if load_icgem_gdf reads an ICGEM file with height column"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample-with-height.gdf")
    icgem_grd = load_icgem_gdf(fname)

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    true_data = np.array([np.arange(nlon)] * nlat, dtype="float64")
    glon, glat = np.meshgrid(lon, lat)
    height = glon + glat

    assert icgem_grd.dims["latitude"] == nlat
    assert icgem_grd.dims["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.h_over_geoid.values)


def test_load_icgem_gdf_usecols():
    "Check if load_icgem_gdf loads ICGEM file reading only first two columns"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    icgem_grd = load_icgem_gdf(fname, usecols=[0, 1])

    s, n, w, e = 16, 28, 150, 164
    nlat, nlon = 7, 8
    shape = (nlat, nlon)
    lat = np.linspace(s, n, nlat, dtype="float64")
    lon = np.linspace(w, e, nlon, dtype="float64")
    height = 1100 * np.ones(shape)

    assert icgem_grd.dims["latitude"] == nlat
    assert icgem_grd.dims["longitude"] == nlon
    npt.assert_equal(icgem_grd.longitude.values, lon)
    npt.assert_equal(icgem_grd.latitude.values, lat)
    npt.assert_allclose(height, icgem_grd.height_over_ell.values)
    assert len(icgem_grd.data_vars) == 1


def test_missing_shape(tmpdir):
    "ICGEM file with missing shape"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    attributes = ["latitude_parallels", "longitude_parallels"]
    for attribute in attributes:
        corrupt = str(tmpdir.join("missing_shape_" + attribute + ".gdf"))
        with open(fname) as gdf_file:
            with open(corrupt, "w") as corrupt_gdf:
                for line in gdf_file:
                    if attribute in line:
                        continue
                    corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_missing_size(tmpdir):
    "ICGEM file with missing size"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = str(tmpdir.join("missing_size.gdf"))
    attribute = "number_of_gridpoints"
    with open(fname) as gdf_file:
        with open(corrupt, "w") as corrupt_gdf:
            for line in gdf_file:
                if attribute in line:
                    continue
                corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_corrupt_shape(tmpdir):
    "ICGEM file with corrupt shape"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    attributes = ["latitude_parallels", "longitude_parallels"]
    for attribute in attributes:
        corrupt = str(tmpdir.join("missing_shape_" + attribute + ".gdf"))
        with open(fname) as gdf_file:
            with open(corrupt, "w") as corrupt_gdf:
                for line in gdf_file:
                    if attribute in line:
                        new_value = int(line.split()[1]) + 1
                        new_line = attribute + "\t" + str(new_value) + "\n"
                        corrupt_gdf.write(new_line)
                    else:
                        corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_missing_cols_names(tmpdir):
    "ICGEM file with missing cols names"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = str(tmpdir.join("missing_cols_names.gdf"))
    with open(fname) as gdf_file:
        with open(corrupt, "w") as corrupt_gdf:
            for line in gdf_file:
                if "latitude" in line and "longitude" in line:
                    continue
                corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_units(tmpdir):
    "ICGEM file with missing units"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = str(tmpdir.join("missing_units.gdf"))
    with open(fname) as gdf_file:
        with open(corrupt, "w") as corrupt_gdf:
            for line in gdf_file:
                if "[mgal]" in line:
                    continue
                corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_empty_line(tmpdir):
    "ICGEM file with missing empty line"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = str(tmpdir.join("missing_empty_line.gdf"))
    with open(fname) as gdf_file:
        with open(corrupt, "w") as corrupt_gdf:
            for line in gdf_file:
                if not line.strip():
                    continue
                corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_attribute(tmpdir):
    "ICGEM file with one missing attribute (not missing unit)"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = str(tmpdir.join("missing_attribute.gdf"))
    with open(fname) as gdf_file:
        with open(corrupt, "w") as corrupt_gdf:
            for line in gdf_file:
                if "longitude" in line and "latitude" in line:
                    parts = line.strip().split()
                    corrupt_gdf.write("\t".join(parts[:2]) + "\n")
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_lat_lon_attributes(tmpdir):
    "ICGEM file with missing longitude or latitude attribute"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    attributes = ["longitude", "latitude"]
    for attribute in attributes:
        corrupt = str(tmpdir.join("missing_" + attribute + "_attribute.gdf"))
        with open(fname) as gdf_file:
            with open(corrupt, "w") as corrupt_gdf:
                for line in gdf_file:
                    if "longitude" in line and "latitude" in line:
                        new_line = line.replace(attribute, "corrupt")
                        corrupt_gdf.write(new_line)
                    else:
                        corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_diff_attrs_vs_cols(tmpdir):
    "ICGEM file with different number of cols vs number of attributes"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = str(tmpdir.join("diff_attributes_vs_cols.gdf"))
    with open(fname) as gdf_file:
        with open(corrupt, "w") as corrupt_gdf:
            for line in gdf_file:
                if "longitude" in line and "latitude" in line:
                    parts = line.strip().split()
                    corrupt_gdf.write("\t".join(parts[:2]) + "\n")
                elif "[mgal]" in line:
                    parts = line.strip().split()
                    corrupt_gdf.write("\t".join(parts[:2]) + "\n")
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_area(tmpdir):
    "ICGEM file with missing area coordinates"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    attributes = [
        "latlimit_north",
        "latlimit_south",
        "longlimit_west",
        "longlimit_east",
    ]
    for attribute in attributes:
        corrupt = str(tmpdir.join("missing_" + attribute + ".gdf"))
        with open(fname) as gdf_file:
            with open(corrupt, "w") as corrupt_gdf:
                for line in gdf_file:
                    if attribute in line:
                        continue
                    corrupt_gdf.write(line)
        with raises(IOError):
            load_icgem_gdf(corrupt)


def test_corrupt_area(tmpdir):
    "ICGEM file with area in header mismatch area from data"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    attributes = [
        "latlimit_north",
        "latlimit_south",
        "longlimit_west",
        "longlimit_east",
    ]
    for attribute in attributes:
        corrupt = str(tmpdir.join("corrupt_" + attribute + ".gdf"))
        with open(fname) as gdf_file:
            with open(corrupt, "w") as corrupt_gdf:
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


def test_empty_file(tmpdir, recwarn):
    "Empty ICGEM file"
    empty_fname = str(tmpdir.join("empty.gdf"))
    with open(empty_fname, "w") as gdf_file:
        gdf_file.write("")
    with raises(IOError):
        load_icgem_gdf(empty_fname)
    assert len(recwarn) == 1
    warning = recwarn.pop()
    assert warning.category == UserWarning
    assert "Empty input file" in str(warning.message)
