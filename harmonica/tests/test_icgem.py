import os
import numpy as np
import numpy.testing as npt

from .. import load_icgem_gdf

MODULE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MODULE_DIR, 'data')


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

    assert icgem_grd.dims['northing'] == nlat
    assert icgem_grd.dims['easting'] == nlon
    npt.assert_equal(icgem_grd.lon.values, lon)
    npt.assert_equal(icgem_grd.lat.values, lat)
    npt.assert_allclose(true_data, icgem_grd.sample_data.values)
    npt.assert_allclose(height, icgem_grd.height.values)
