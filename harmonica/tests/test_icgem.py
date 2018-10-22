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
    "Check if load_icgem_gdf reads ICGEM test data correctly"
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


def test_corrupt_icgem_gdf(tmpdir):
    "Check if load_icgem_gdf detects a corrupt ICGEM gdf file"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")

    # Missing shape
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

    # Missing size
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

    # Corrupted shape vs size
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
