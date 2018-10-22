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
    lon, lat = np.meshgrid(lon, lat)
    true_data = np.array([np.arange(nlon)] * nlat, dtype="float64")
    height = np.zeros(shape)

    assert icgem_grd.dims["northing"] == nlat
    assert icgem_grd.dims["easting"] == nlon
    npt.assert_equal(icgem_grd.lon.values, lon)
    npt.assert_equal(icgem_grd.lat.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.height.values)


def test_missing_shape(tmpdir):
    "Check if load_icgem_gdf detects an ICGEM file with missing shape"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("missing_shape.gdf")
    attribute = "latitude_parallels"
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if attribute in line:
                    continue
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_size(tmpdir):
    "Check if load_icgem_gdf detects an ICGEM file with missing size"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("missing_size.gdf")
    attribute = "number_of_gridpoints"
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if attribute in line:
                    continue
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_corrupt_shape(tmpdir):
    "Check if load_icgem_gdf detects an corrupt ICGEM file with corrupt shape"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("corrupt_shape.gdf")
    attribute = "latitude_parallels"
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if attribute in line:
                    new_value = int(line.split()[1]) + 1
                    new_line = attribute + "\t" + str(new_value) + "\n"
                    corrupt_gdf.write(new_line)
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_cols_names(tmpdir):
    "Check if load_icgem_gdf detects ICGEM file with missing cols names"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("missing_cols_names.gdf")
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if "latitude" in line and "longitude" in line:
                    continue
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_units(tmpdir):
    "Check if load_icgem_gdf detects an ICGEM file with missing units"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("missing_units.gdf")
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if "[mgal]" in line:
                    continue
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_missing_empty_line(tmpdir):
    "Check if load_icgem_gdf detects an ICGEM file with missing empty line"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("missing_empty_line.gdf")
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if not line.strip():
                    continue
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_corrupt_attributes(tmpdir):
    "Check if load_icgem_gdf detects an ICGEM file with one missing attr"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("corrupt_attributes.gdf")
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
                if "longitude" in line and "latitude" in line:
                    parts = line.strip().split()
                    corrupt_gdf.write("\t".join(parts[:2]) + "\n")
                else:
                    corrupt_gdf.write(line)
    with raises(IOError):
        load_icgem_gdf(corrupt)


def test_corrupt_attrs_vs_cols(tmpdir):
    "ICGEM file with different number of cols vs number of attributes"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    corrupt = tmpdir.join("corrupt_attributes.gdf")
    with open(fname) as f:
        with open(corrupt, "w") as corrupt_gdf:
            for line in f:
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
