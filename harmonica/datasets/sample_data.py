"""
Functions to load sample datasets used in the Harmonica docs.
"""
import os
import tempfile
import lzma
import shutil

import xarray as xr
import pandas as pd
import pooch

from ..version import full_version

POOCH = pooch.create(
    path=["~", ".harmonica", "data"],
    base_url="https://github.com/fatiando/harmonica/raw/{version}/data/",
    version=full_version,
    version_dev="master",
    env="HARMONICA_DATA_DIR",
)
POOCH.load_registry(os.path.join(os.path.dirname(__file__), "registry.txt"))


def fetch_gravity_earth():
    """
    Fetch a global grid of Earth gravity.

    Gravity is the magnitude of the gravity vector of the Earth (gravitational +
    centrifugal). The gravity observations are at 10 km (geometric) height and on a
    regular grid with 0.5 degree spacing. The grid was generated from the spherical
    harmonic model EIGEN-6C4 [Forste_etal2014]_ using the `ICGEM Calculation Service
    <http://icgem.gfz-potsdam.de/>`__. See the ``attrs`` attribute of the
    :class:`xarray.Dataset` for information regarding the grid generation.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        The gravity grid (in mGal). Includes a computation height grid
        (``height_over_ell``). Coordinates are latitude and longitude.

    """
    fname = POOCH.fetch("gravity-earth-0.5deg.nc.xz")
    # The heights are stored as ints and data as float32 to save space on the data file.
    # Cast them to float64 to avoid integer division errors.
    data = _load_xz_compressed_grid(fname, engine="scipy").astype("float64")
    return data


def fetch_topography_earth():
    """
    Fetch a global grid of Earth relief (topography and bathymetry).

    The grid is based on the ETOPO1 model [AmanteEakins2009]_. The original model has 1
    arc-minute grid spacing but here we downsampled to 0.5 degree grid spacing to save
    space and download times. The downsampled grid was generated from a spherical
    harmonic model using the `ICGEM Calculation Service
    <http://icgem.gfz-potsdam.de/>`__. See the ``attrs`` attribute of the returned
    :class:`xarray.Dataset` for information regarding the grid generation.

    ETOPO1 heights are referenced to "sea level".

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        The topography grid (in meters). Coordinates are latitude and longitude.

    """
    fname = POOCH.fetch("etopo1-0.5deg.nc.xz")
    # The data are stored as int16 to save disk space. Cast them to floats to avoid
    # integer division problems when processing.
    data = _load_xz_compressed_grid(fname, engine="scipy").astype("float64")
    return data


def fetch_rio_magnetic():
    """
    Fetch total-field magnetic anomaly data from Rio de Janeiro, Brazil.

    These data are a subsection of an airborne survey of the state of Rio de Janeiro,
    Brazil, conducted in 1978. The data are made available by the Geological Survey of
    Brazil (CPRM) through their `GEOSGB portal <http://geosgb.cprm.gov.br/>`__.

    The columns of the data table are longitude, latitude, total-field magnetic anomaly
    (nanoTesla), observation height above the WGS84 ellipsoid (in meters), flight line
    type (LINE or TIE), and flight line number for each data point.

    The anomaly is calculated with respect to the IGRF field parameters listed on the
    table below. See the original data for more processing information.

    +----------+-----------+----------------+-------------+-------------+
    |               IGRF for year 1978.3 at 500 m height                |
    +----------+-----------+----------------+-------------+-------------+
    | Latitude | Longitude | Intensity (nT) | Declination | Inclination |
    +==========+===========+================+=============+=============+
    |  -22º15' |  -42º15'  |     23834      |   -19º19'   |   -27º33'   |
    +----------+-----------+----------------+-------------+-------------+

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        The magnetic anomaly data.

    """
    return pd.read_csv(POOCH.fetch("rio-magnetic.csv.xz"), compression="xz")


def _load_xz_compressed_grid(fname, **kwargs):
    """
    Load a netCDF grid that has been xz compressed. Keyword arguments are passed to
    :func:`xarray.open_dataset`.
    """
    decompressed = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    try:
        with decompressed:
            with lzma.open(fname, "rb") as compressed:
                shutil.copyfileobj(compressed, decompressed)
        # Call load to make sure the data are loaded into memory and not linked to file
        grid = xr.open_dataset(decompressed.name, **kwargs).load()
        # Close any files associated with this dataset to make sure can delete them
        grid.close()
    finally:
        os.remove(decompressed.name)
    return grid
