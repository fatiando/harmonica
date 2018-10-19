import numpy as np
import xarray as xr


def load_icgem_gdf(fname, **kwargs):
    """
    Reads data from an ICGEM .gdf file.

    ICGEM Calculation Service generates gravity fields grids from
    spherical harmonic models.

    http://icgem.gfz-potsdam.de/home

    Parameters:

    * fname: string
        Name of the ICGEM .gdf file
    * **kwargs:
        Arguments that will be passed to `numpy.loadtxt`.

    Returns:
    * icgem_grd : xarray.Dataset
        An `xarray.Dataset` with the data from the file.
        The header of the gdf file is passed into the `attr` argument
        of `xarray.Dataset`.
    """
    if 'usecols' not in kwargs:
        kwargs['usecols'] = None
    with open(fname) as f:
        # Read the header and extract metadata
        metadata = {}
        metadata_line = True
        shape = [None, None]
        size = None
        height = None
        attributes = None
        attr_line = False
        area = [None] * 4
        for line in f:
            if line.strip()[:11] == 'end_of_head':
                break
            if not line.strip():
                attr_line = True
                metadata_line = False
                continue
            if line.strip() and metadata_line:
                metadata[line.split()[0]] = ''.join(line.split()[1:])
            if not attr_line:
                parts = line.strip().split()
                if parts[0] == 'height_over_ell':
                    height = float(parts[1])
                elif parts[0] == 'latitude_parallels':
                    shape[0] = int(parts[1])
                elif parts[0] == 'longitude_parallels':
                    shape[1] = int(parts[1])
                elif parts[0] == 'number_of_gridpoints':
                    size = int(parts[1])
                elif parts[0] == 'latlimit_south':
                    area[0] = float(parts[1])
                elif parts[0] == 'latlimit_north':
                    area[1] = float(parts[1])
                elif parts[0] == 'longlimit_west':
                    area[2] = float(parts[1])
                elif parts[0] == 'longlimit_east':
                    area[3] = float(parts[1])
            else:
                attributes = line.strip().split()
                attr_line = False
        # Read the numerical values
        rawdata = np.loadtxt(f, ndmin=2, unpack=True, **kwargs)

    # Sanity checks
    if not all(n is not None for n in shape):
        raise IOError("Couldn't read shape of grid.")
    if size is None:
        raise IOError("Couldn't read size of grid.")
    shape = tuple(shape)
    if shape[0] * shape[1] != size:
        raise IOError(
            "Grid shape '{}' and size '{}' mismatch.".format(shape, size))
    if attributes is None:
        raise IOError("Couldn't read column names.")
    if kwargs['usecols'] is not None:
        attributes = [attributes[i] for i in kwargs['usecols']]
    if len(attributes) != rawdata.shape[0]:
        raise IOError(
            "Number of attributes ({}) and data columns ({}) mismatch".format(
                len(attributes), rawdata.shape[0]))
    if not all(i is not None for i in area):
        raise IOError("Couldn't read the grid area.")
    if 'latitude' not in attributes:
        raise IOError("Couldn't find latitude column.")
    if 'longitude' not in attributes:
        raise IOError("Couldn't find longitude column.")

    # Create xarray.Dataset
    icgem_grd = xr.Dataset()
    icgem_grd.attrs = metadata
    for attr, value in zip(attributes, rawdata):
        # Need to invert the data matrices in latitude "[::-1]"
        # because the ICGEM grid gets varies latitude from N to S
        value = value.reshape(shape)[::-1]
        if attr == 'latitude':
            icgem_grd.coords['lat'] = (('northing', 'easting'), value)
        elif attr == 'longitude':
            icgem_grd.coords['lon'] = (('northing', 'easting'), value)
        else:
            icgem_grd[attr] = (('northing', 'easting'), value)
    if (height is not None) and ('height' not in attributes):
        icgem_grd['height'] = (('northing', 'easting'), height * np.ones(shape))

    # Check area from header equals to area from data in cols
    area_from_cols = (icgem_grd.lat.values.min(),
                      icgem_grd.lat.values.max(),
                      icgem_grd.lon.values.min(),
                      icgem_grd.lon.values.max())
    if not np.allclose(area, area_from_cols):
        errline = "Grid area read ({}) and calculated from attributes " + \
                  "({}) mismatch.".format(area, area_from_cols)
        raise IOError(errline)
    return icgem_grd
