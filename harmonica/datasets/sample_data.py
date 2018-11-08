"""
Functions to load sample datasets used in the Harmonica docs.
"""
import os
import tempfile
import lzma
import shutil

import xarray as xr
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
    data = _load_xz_compressed_grid(fname, engine="scipy")
    return data


def fetch_topography_earth():
    """
    Fetch a global grid of Earth relief (topography and bathymetry).

    The grid is based on the ETOPO1 model [AmanteEakins2009]_. The original model has 1
    arc-minute grid spacing but here we downsampled to 0.5 degree grid spacing to save
    space and download times. The downsampled grid was generated from a spherical
    harmonic model using the `ICGEM Calculation Service
    <http://icgem.gfz-potsdam.de/>`__. See the ``attrs`` attribute of the
    :class:`xarray.Dataset` for information regarding the grid generation.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        The topography grid (in meters). Coordinates are latitude and longitude.

    """
    fname = POOCH.fetch("etopo1-0.5deg.nc.xz")
    data = _load_xz_compressed_grid(fname, engine="scipy")
    return data


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
