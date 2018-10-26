"""
Functions for interacting with standarized data files that contain gravity or
magnetic geophysical data.
"""
import numpy as np
import xarray as xr


def load_icgem_gdf(fname, **kwargs):
    """
    Reads data from an ICGEM .gdf file.

    The `ICGEM Calculation Service
    <http://icgem.gfz-potsdam.de/>`__[BarthelmesKohler2016]_
    generates gravity field grids from spherical harmonic models.
    They use a custom ASCII grid format with information in the header.
    This function can read the format and parse information from the header.
    It returns the data in a :class:`xarray.Dataset` for convenience and
    reduced storage requirements.

    Parameters
    ----------
    fname : string
        Name of the ICGEM .gdf file
    **kwargs
        Extra keyword arguments to this function will be passed to
        :func:`numpy.loadtxt`.

    Returns
    -------
    icgem_ds : :class`xarray.Dataset`
        An :class`xarray.Dataset` with the data from the file.
        The header of the gdf file is passed into the ``attr`` argument
        of the :class`xarray.Dataset`.
    """
    if "usecols" not in kwargs:
        kwargs["usecols"] = None
    rawdata, metadata = _read_gdf_file(fname, **kwargs)
    shape = (metadata["latitude_parallels"], metadata["longitude_parallels"])
    area = [
        metadata["latlimit_south"],
        metadata["latlimit_north"],
        metadata["longlimit_west"],
        metadata["longlimit_east"],
    ]
    attributes = metadata["attributes"]
    if kwargs["usecols"] is not None:
        attributes = [attributes[i] for i in kwargs["usecols"]]
    if len(attributes) != rawdata.shape[0]:
        raise IOError(
            "Number of attributes ({}) and data columns ({}) mismatch".format(
                len(attributes), rawdata.shape[0]
            )
        )

    # Create xarray.Dataset
    icgem_ds = xr.Dataset()
    icgem_ds.attrs = metadata
    for attr, value in zip(attributes, rawdata):
        # Need to invert the data matrices in latitude "[::-1]"
        # because the ICGEM grid gets varies latitude from N to S
        value = value.reshape(shape)[::-1]
        if attr == "latitude":
            icgem_ds.coords["lat"] = (("northing", "easting"), value)
        elif attr == "longitude":
            icgem_ds.coords["lon"] = (("northing", "easting"), value)
        else:
            icgem_ds[attr] = (("northing", "easting"), value)
    if "height_over_ell" in metadata and "height" not in attributes:
        height = float(metadata["height_over_ell"].split()[0])
        icgem_ds["height"] = (("northing", "easting"), height * np.ones(shape))

    # Check area from header equals to area from data in cols
    area_from_cols = (
        icgem_ds.lat.values.min(),
        icgem_ds.lat.values.max(),
        icgem_ds.lon.values.min(),
        icgem_ds.lon.values.max(),
    )
    if not np.allclose(area, area_from_cols):
        errline = (
            "Grid area read ({}) and calculated from attributes "
            "({}) mismatch.".format(area, area_from_cols)
        )
        raise IOError(errline)
    return icgem_ds


def _read_gdf_file(fname, **kwargs):
    "Read ICGEM gdf file and returns metadata dict and data in cols as np.array"
    with open(fname) as gdf_file:
        # Read the header and extract metadata
        metadata = {}
        metadata_line = True
        attributes_units_line = False
        for line in gdf_file:
            if line.strip()[:11] == "end_of_head":
                break
            if not line.strip():
                metadata_line = False
                continue
            if metadata_line:
                parts = line.strip().split()
                metadata[parts[0]] = " ".join(parts[1:])
            else:
                if not attributes_units_line:
                    metadata["attributes"] = line.strip().split()
                    attributes_units_line = True
                else:
                    metadata["attributes_units"] = line.strip().split()
        # Read the numerical values
        rawdata = np.loadtxt(gdf_file, ndmin=2, unpack=True, **kwargs)
    _check_integrity(metadata)
    return rawdata, metadata


def _check_integrity(metadata):
    "Check the integrity of ICGEM gdf file metadata."
    needed_args = [
        "latitude_parallels",
        "longitude_parallels",
        "number_of_gridpoints",
        "latlimit_south",
        "latlimit_north",
        "longlimit_west",
        "longlimit_east",
    ]
    # Check for needed arguments in metadata dictionary
    for arg in needed_args:
        if arg in metadata:
            if "limit" in arg:
                metadata[arg] = float(metadata[arg].split()[0])
            else:
                metadata[arg] = int(metadata[arg].split()[0])
        else:
            raise IOError("Couldn't read {} field from gdf file header".format(arg))
    if "attributes" not in metadata:
        raise IOError("Couldn't read column names.")
    if "attributes_units" not in metadata:
        raise IOError("Couldn't read column units.")
    # Check cols names and units integrity
    if len(metadata["attributes"]) != len(metadata["attributes_units"]):
        raise IOError(
            "Number of attributes ({}) and units ({}) mismatch".format(
                len(metadata["attributes"]), len(metadata["attributes_units"])
            )
        )
    for arg in ["latitude", "longitude"]:
        if arg not in metadata["attributes"]:
            raise IOError("Couldn't find {} column.".format(arg))
    # Check proper values for shape and size
    shape = (metadata["latitude_parallels"], metadata["longitude_parallels"])
    size = metadata["number_of_gridpoints"]
    if shape[0] * shape[1] != size:
        raise IOError("Grid shape '{}' and size '{}' mismatch.".format(shape, size))
