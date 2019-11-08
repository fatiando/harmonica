"""
Functions for interacting with standarized data files that contain gravity or
magnetic geophysical data.
"""
import numpy as np
import xarray as xr


def load_icgem_gdf(fname, **kwargs):
    """
    Reads data from an ICGEM .gdf file.

    The `ICGEM Calculation Service <http://icgem.gfz-potsdam.de/>`__
    [BarthelmesKohler2016]_ generates gravity field grids from spherical
    harmonic models. They use a custom ASCII grid format with information in
    the header. This function can read the format and parse information from
    the header. It returns the data in a :class:`xarray.Dataset` for
    convenience and reduced storage requirements.

    Parameters
    ----------
    fname : string
        Name of the ICGEM .gdf file
    kwargs
        Extra keyword arguments to this function will be passed to
        :func:`numpy.loadtxt`.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        An :class:`xarray.Dataset` with the data from the file.
        The header of the gdf file is available through the ``attr`` attribute
        of the :class:`xarray.Dataset`.

    """
    rawdata, metadata = _read_gdf_file(fname, **kwargs)
    shape = (int(metadata["latitude_parallels"]), int(metadata["longitude_parallels"]))
    data = dict(zip(metadata["attributes"], rawdata))
    coords = {
        "longitude": data["longitude"].reshape(shape)[0, :],
        "latitude": data["latitude"].reshape(shape)[:, 0][::-1],
    }
    dims = ["latitude", "longitude"]
    data_vars = {
        name: (dims, value.reshape(shape)[::-1])
        for name, value in data.items()
        if name not in dims
    }
    # If the grid is at constant height, add the height as a matrix for
    # convenience (otherwise, would have to parse from the attrs)
    if "height_over_ell" in metadata:
        height = float(metadata["height_over_ell"].split()[0])
        data_vars["height_over_ell"] = (dims, np.full(shape, height))
    # Can't have lists in the Dataset metadata to make it compatible with
    # netCDF3. This way the data can be saved using only scipy as a dependency
    # instead of netcdf4.
    metadata["attributes"] = " ".join(metadata["attributes"])
    metadata["attributes_units"] = " ".join(metadata["attributes_units"])
    grid = xr.Dataset(data_vars, coords=coords, attrs=metadata)
    # Check area from header equals to area from data in cols
    area = [
        float(metadata["latlimit_south"]),
        float(metadata["latlimit_north"]),
        float(metadata["longlimit_west"]),
        float(metadata["longlimit_east"]),
    ]
    area_from_cols = (
        grid.latitude.values.min(),
        grid.latitude.values.max(),
        grid.longitude.values.min(),
        grid.longitude.values.max(),
    )
    if not np.allclose(area, area_from_cols):
        raise IOError(
            "Grid area read ({}) and calculated from attributes "
            "({}) mismatch.".format(area, area_from_cols)
        )
    return grid


def _read_gdf_file(fname, **kwargs):
    """
    Read ICGEM gdf file and returns metadata dict and data in cols as np.array
    """
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
    _check_gdf_integrity(metadata)
    # Remove column names from the metadata if they weren't read
    if kwargs.get("usecols", None) is not None:
        metadata["attributes"] = [metadata["attributes"][i] for i in kwargs["usecols"]]
    if len(metadata["attributes"]) != rawdata.shape[0]:
        raise IOError(
            "Number of attributes ({}) and data columns ({}) mismatch".format(
                len(metadata["attributes"]), rawdata.shape[0]
            )
        )
    return rawdata, metadata


def _check_gdf_integrity(metadata):
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
            metadata[arg] = metadata[arg].split()[0]
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
    metadata["attributes_units"] = [
        attr.replace("[", "").replace("]", "").strip()
        for attr in metadata["attributes_units"]
    ]
    for arg in ["latitude", "longitude"]:
        if arg not in metadata["attributes"]:
            raise IOError("Couldn't find {} column.".format(arg))
    # Check proper values for shape and size
    shape = (int(metadata["latitude_parallels"]), int(metadata["longitude_parallels"]))
    size = int(metadata["number_of_gridpoints"])
    if shape[0] * shape[1] != size:
        raise IOError("Grid shape '{}' and size '{}' mismatch.".format(shape, size))
