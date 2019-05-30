"""
Meshes and layers of simple geometric elements.
"""
import numpy as np
import xarray as xr
from verde.coordinates import spacing_to_shape

from . import geodetic_to_spherical


def tesseroid_layer(
    region,
    spacing=None,
    shape=None,
    adjust="spacing",
    top=None,
    bottom=None,
    density=None,
    coordinates="spherical",
    region_centers=True,
):
    """
    Create a layer of tesseroids.

    The layer can be specified by either the number of tesseroids in each dimension (the
    *shape*) or by the size of each tesseroid (the *spacing*).

    The layer will be a :class:`xarray.Dataset` with ``longitude`` and ``latitude``
    coordinates and `top`, `bottom` and `density` as data.
    The ``longitude`` and ``latitude`` will be the coordinates of the center of each
    tesseroid on the layer in geocentric spherical coordinates.
    The ``top`` and ``bottom`` parameters can be defined either in a geocentric
    spherical or in a geodetic coordinate system.

    .. warning :
        If ``top`` and ``bottom`` parameters are defined under a geodetic coordinate
        system, they will be converted to a spherical geocentric one.
        Because tesseroids cannot be defined on a geodetic coordinate system, the
        latitude coordinates will remain unchanged.

    If the given region is not divisible by the desired spacing, either the region or
    the spacing will have to be adjusted. By default, the spacing will be rounded to the
    nearest multiple. Optionally, the East and North boundaries of the region can be
    adjusted to fit the exact spacing given. See the examples below.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in geocentric spherical coordinates.
    spacing : float, tuple = (d_lat, d_lon), or None
        The latitudinal and longitudinal spacing between the center of neighbour
        tesseroids, respectively, i.e. the longitudinal and latitudinal size of each
        tesseroid. A single value means that the spacing is equal in both directions.
    shape : tuple = (n_lat, n_lon) or None
        The number of tesseroids in the latitudinal and longitudinal directions,
        respectively.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if *shape* is
        given instead of *spacing*. Defaults to adjusting the spacing.
    top : float or None
        Coordinate of outer surface of the layer in geocentric spherical or geodetic
        coordinates. If ``None`` a ``np.nan`` array will be added to the
        :class:``xarray.Dataset``.
    bottom : float or None
        Coordinate of inner surface of the layer in geocentric spherical or geodetic
        coordinates. If ``None`` a ``np.nan`` array will be added to the
        :class:``xarray.Dataset``.
    density : float or None
        Density of the tesseroids on the layer. If ``None`` a zeroes array will be
        added to the :class:``xarray.Dataset``.
    coordinates : {"spherical", "geodetic"}
        Specify under which coordinate system the ``top`` and ``bottom`` parameters are
        defined. Default is ``spherical``.
    region_centers : bool
        If True, the region coordinates are assumed to be the center of the
        tesseroids located on the boundaries of the layer. If False, the region
        boundaries are considered as the longitudinal and latitudinal bounds of the
        extreme tesseroids. In practice this means that there won't be any portion of
        mass outside the given region. Default is True.

    Returns
    -------
    tesseroid_layer : :class:``xarray.Dataset``
        Dataset containing the coordinates of the tesseroids' centers in geocentric
        spherical coordinates along with the top and bottom coordinates and density of
        each tesseroid.
    """
    # check_region(region)
    if shape is not None and spacing is not None:
        raise ValueError("Both shape and spacing provided. Only one is allowed.")
    if shape is None and spacing is None:
        raise ValueError("Either a shape or a spacing must be provided.")
    if spacing is not None:
        shape, region = spacing_to_shape(region, spacing, adjust)
    if coordinates not in ("spherical", "geodetic"):
        raise ValueError(
            "Invalid geographic coordinate system '{}'.".format(coordinates)
            + " Should be 'spherical' or 'geodetic'."
        )
    longitude = np.linspace(region[0], region[1], shape[1])
    latitude = np.linspace(region[2], region[3], shape[0])
    coords = {"longitude": longitude, "latitude": latitude}
    if top is None:
        top = np.nan
    if bottom is None:
        bottom = np.nan
    if density is None:
        density = 0
    dims = ("latitude", "longitude")
    data_vars = {
        "top": (dims, top * np.ones(shape)),
        "bottom": (dims, bottom * np.ones(shape)),
        "density": (dims, density * np.ones(shape)),
    }
    layer = xr.Dataset(data_vars, coords=coords)
    # Convert top and bottom coordinates if given in geodetic
    if coordinates == "geodetic":
        if top is not None:
            _, _, layer["top"] = geodetic_to_spherical(
                layer.longitude, layer.latitude, layer.top
            )
        if bottom is not None:
            _, _, layer["bottom"] = geodetic_to_spherical(
                layer.longitude, layer.latitude, layer.bottom
            )
    return layer
