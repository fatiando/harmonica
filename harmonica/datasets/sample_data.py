"""
Functions to load sample datasets used in the Harmonica docs.
"""
import pkg_resources
import xarray as xr
import pandas as pd
import pooch

from ..version import full_version

REGISTRY = pooch.create(
    path=pooch.os_cache("harmonica"),
    base_url="https://github.com/fatiando/harmonica/raw/{version}/data/",
    version=full_version,
    version_dev="master",
    env="HARMONICA_DATA_DIR",
)
with pkg_resources.resource_stream(
    "harmonica.datasets", "registry.txt"
) as registry_file:
    REGISTRY.load_registry(registry_file)


def locate():
    r"""
    The absolute path to the sample data storage location on disk.

    This is where the data are saved on your computer. The location is
    dependent on the operating system. The folder locations are defined by the
    ``appdirs``  package (see the `appdirs documentation
    <https://github.com/ActiveState/appdirs>`__).

    The location can be overwritten by the ``HARMONICA_DATA_DIR`` environment
    variable to the desired destination.

    Returns
    -------
    path : str
        The local data storage location.

    """
    return str(REGISTRY.abspath)


def fetch_geoid_earth():
    """
    Fetch a global grid of the geoid height.

    The geoid height is the height of the geoid above (positive) or below
    (negative) the ellipsoid (WGS84). The data are on a regular grid with 0.5
    degree spacing, which was generated from the spherical harmonic model
    EIGEN-6C4 [Forste_etal2014]_ using the `ICGEM Calculation Service
    <http://icgem.gfz-potsdam.de/>`__. See the ``attrs`` attribute of the
    :class:`xarray.Dataset` for information regarding the grid generation.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        The geoid grid (in meters). Coordinates are geodetic latitude and
        longitude.

    """
    fname = REGISTRY.fetch("geoid-earth-0.5deg.nc.xz", processor=pooch.Decompress())
    data = xr.open_dataset(fname, engine="scipy").astype("float64")
    return data


def fetch_gravity_earth():
    """
    Fetch a global grid of Earth gravity.

    Gravity is the magnitude of the gravity vector of the Earth (gravitational
    + centrifugal). The gravity observations are at 10 km (geometric) height
    and on a regular grid with 0.5 degree spacing. The grid was generated from
    the spherical harmonic model EIGEN-6C4 [Forste_etal2014]_ using the `ICGEM
    Calculation Service <http://icgem.gfz-potsdam.de/>`__. See the ``attrs``
    attribute of the :class:`xarray.Dataset` for information regarding the grid
    generation.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        The gravity grid (in mGal). Includes a computation (geometric) height
        grid (``height_over_ell``). Coordinates are geodetic latitude and
        longitude.

    """
    fname = REGISTRY.fetch("gravity-earth-0.5deg.nc.xz", processor=pooch.Decompress())
    # The heights are stored as ints and data as float32 to save space on the
    # data file. Cast them to float64 to avoid integer division errors.
    data = xr.open_dataset(fname, engine="scipy").astype("float64")
    return data


def fetch_topography_earth():
    """
    Fetch a global grid of Earth relief (topography and bathymetry).

    The grid is based on the ETOPO1 model [AmanteEakins2009]_. The original
    model has 1 arc-minute grid spacing but here we downsampled to 0.5 degree
    grid spacing to save space and download times. The downsampled grid was
    generated from a spherical harmonic model using the `ICGEM Calculation
    Service <http://icgem.gfz-potsdam.de/>`__. See the ``attrs`` attribute of
    the returned :class:`xarray.Dataset` for information regarding the grid
    generation.

    ETOPO1 heights are referenced to "sea level".

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        The topography grid (in meters) relative to sea level. Coordinates are
        geodetic latitude and longitude.

    """
    fname = REGISTRY.fetch("etopo1-0.5deg.nc.xz", processor=pooch.Decompress())
    # The data are stored as int16 to save disk space. Cast them to floats to
    # avoid integer division problems when processing.
    data = xr.open_dataset(fname, engine="scipy").astype("float64")
    return data


def fetch_britain_magnetic():
    """
    Fetch total-field magnetic anomaly data of Great Britain.

    These data are a complete airborne survey of the entire Great Britain
    conducted between 1955 and 1965. The data are made available by the
    British Geological Survey (BGS) through their `geophysical data portal
    <https://www.bgs.ac.uk/products/geophysics/aeromagneticRegional.html>`__.

    License: `Open Government License
    <http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>`__

    The columns of the data table are longitude, latitude, total-field magnetic
    anomaly (nanoTesla), observation height relative to the WGS84 datum (in
    meters), survey area, and line number and line segment for each data point.

    Latitude, longitude, and elevation data converted from original OSGB36
    (epsg:27700) coordinate system to WGS84 (epsg:4326) using to_crs function
    in GeoPandas.

    See the original data for more processing information.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        The magnetic anomaly data.
    """
    return pd.read_csv(REGISTRY.fetch("britain-magnetic.csv.xz"), compression="xz")


def fetch_south_africa_gravity():
    """
    Fetch gravity station data from South Africa

    This data base (14559 records), received in January 1986, consists in land
    gravity surveys within the boundaries of the Republic of South Africa. The
    survey was conducted by Dr. R.J. Kleywegt with the contribution of the
    Republic of South Africa, the Department of Mineral and Energy Affairs and
    the Geological Survey. The data was made available by the `National Centers
    for Environmental Information (NCEI) <https://www.ngdc.noaa.gov/>`__
    (formerly NGDC) and are in the `public domain
    <https://www.ngdc.noaa.gov/ngdcinfo/privacy.html#copyright-notice>`__.
    Original data files can be found at:
    https://www.ngdc.noaa.gov/mgg/gravity/1999/data/regional/africa/

    Principal gravity parameters include elevation and observed gravity. The
    observed gravity values are referenced to the International Gravity
    Standardization Net 1971 (IGSN 71). Elevations are referenced above the sea
    level. Both ``longitude`` and ``latitude`` are given in degrees,
    ``elevation`` in meters and ``gravity`` in mGal.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        The gravity data.

    """
    fname = REGISTRY.fetch("south-africa-gravity.ast.xz")
    columns = ["latitude", "longitude", "elevation", "gravity"]
    return pd.read_csv(fname, sep=r"\s+", names=columns, compression="xz")
