# Function to read in GXF files, provided by USGS Generally
# No guarantee this will read in your specific GXF files

# Written by Thomas Martin, Unidata
# Inspired by this gist: https://gist.github.com/jobar8/683483df605a906fb3da747b64627305
# created by Joseph Barraud and made available under the BSD License.

"""
Functions for reading Geosoft GXF (Grid eXchange File) format files.

The GXF format is an ASCII file format for gridded data developed by Geosoft.
For detailed format specification, see:
https://pubs.usgs.gov/of/1999/of99-514/grids/gxf.pdf

Test data provided by USGS:
Kucks, R.P., and Hill, P.L., 2000, Wyoming aeromagnetic and gravity maps and data—
A web site for distribution of data: U.S. Geological Survey Open-File Report 00-0198,
https://pubs.usgs.gov/of/2000/ofr-00-0198/html/wyoming.htm

GXF (Grid eXchange File) is a standard ASCII file format for
exchanging gridded data among different software systems. 
Software that supports the GXF standard will be able to import
properly formatted GXF files and export grids in GXF format.

Grid Description:
A grid is a rectangular array of points at which single data
values define a two dimensional function. Grid point locations
are related to a Grid Coordinate System (GCS), which is a right
handed Cartesian system with X and Y axis defined by the bottom
and left sides of a grid array.  The grid point at the bottom,
left corner of the array is the origin of the GCS.  All distances
are in meters.

GCS coordinates are related to a Base Coordinate System (BCS)
through a plane translation and rotation.  The origin of the GCS
is located at point (x0,y0) in the BCS, and the X and Y grid
indices are related to BCS units through the separation between
points in the GCS X and Y directions.

Labeled Data Objects and Comments

A GXF file is an ASCII file made up of a number of labeled data
objects and comments.  Each labeled data object has a label line
followed by one or more data lines.  A label line is identified
by a '#' character in the first column followed immediately by an
upper-case label.  The data associated with that label are found
on one or more lines that follow the label.

Lines

All lines in a GXF file must be less than or equal to 80
characters in length. Any lines that are not part of a labeled
data object are ignored and can be used to place comments within
a GXF file.  Programs that read GXF files will skip such comment
lines while they search for the next GXF data object.

GXF Object Definitions

#TITLE
A one line descriptive title of the grid.  Some grid formats
include textual descriptions of the grid, and this information
can be placed in a #TITLE object.
Default:        blank title

#POINTS
The number of points in each grid row (horizontal or vertical as
defined by the #SENSE object).
Default:        no default - this object is required.

#ROWS
The number of rows in the grid.  A grid row (or vector) is a
collection of consecutive grid points that represent the grid
values along a horizontal or vertical line in the grid.  The
complete grid is then defined by a consecutive sequence of grid
rows.
Default:        no default - this object is required.

#PTSEPARATION
The separation between points in the grid.  This should be in
Base Coordinate System units (ground units for geographically
based grids).
Default:        1.0

#RWSEPARATION
The separation between rows in the grid.  These should be in Base
Coordinate System units (ground units for geographically based
grids).
Default:        1.0

#XORIGIN 
The X location of the bottom left corner of the grid in the Base
Coordinate System.
Default:        0.0

#YORIGIN
The Y location of the bottom left corner of the grid in the Base
Coordinate System.
Default:        0.0

#ROTATION
The rotation angle of the grid.  This is the counter-clockwise
angle of the bottom edge of the grid with respect to the Base
Coordinate System X axis. Rotation only has meaning for Base
Coordinate Systems that use the same units on the X and Y axis.
Default:        0.0

#SENSE
The first point of the first row of the stored grid can be at any
corner of the grid rectangle, and the grid rows can be run
vertically or horizontally. The SENSE object defines this storage
sense as follows:
        ą1      first point at bottom left of grid
        ą2      first point at upper left of grid
        ą3      first point at upper right of grid
        ą4      first point at bottom right of grid
A positive SENSE stores rows in a right-handed sense; a negative
SENSE stores rows in a left-handed sense.  This means that if you
were standing at the first grid point and looking into the grid,
the first grid row would extend to your right for a right handed
grid (positive sense), or to your left for a left handed sense
(left-handed grid): (All grids on this CD have SENSE=+1.)
Default:        1 (first point at bottom left, rows left to
right)

#TRANSFORM
This keyword is followed by two numbers on the same line:  SCALE
and OFFSET, which are used to transform the grid data to desired
units:
Z = G * SCALE + OFFSET
where
        Z       grid value in the desired unit
        G       are grid values as specified in the #GRID object
Default:        SCALE = 1.0,  OFFSET = 0.0

#DUMMY
The grid must be rectangular (every row must have the same number
of points). The dummy value defined by this object is used to
define blank areas of the grid.  Any grids that include blank
areas must define a dummy value.
Default:        no dummy value.

#GRID
The grid data is listed point by point and row by row.  The #GRID
object and data is always the last object in a GXF file. The
first data point is at the location indicated by #SENSE, and is
followed by successive points in that row of points (either
horizontal or vertical), then the points in the next row, and so
on.  The points in a row can follow on to the next data line,
although each new row must start on a new data line.  A GXF
reading program can expect #ROWS of #POINTS for a total of #ROWS
times  #POINTS data values.
Default: none, must be included as the last object in a
GXF file.

Data for testing is also from the USGS:

Kucks, R.P., and Hill, P.L., 2000, Wyoming aeromagnetic and gravity maps and data—A web site for distribution of data: U.S. Geological Survey Open-File Report 00-0198, https://pubs.usgs.gov/of/2000/ofr-00-0198/html/wyoming.htm

"""

"""
Function to read USGS GXF file
"""

import numpy as np
import xarray as xr
from typing import Tuple, List, Dict, Union, Any

def read_gxf_raw(infile: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Read a GXF file and return raw data list and headers.
    Following official GXF specifications from Geosoft.
    
    Parameters:
    infile (str): Path to the GXF file
    
    Returns:
    tuple: (data_list, headers)
        - data_list: list of raw data strings
        - headers: dictionary of GXF headers
    """
    # Read only header lines until #GRID
    headers: Dict[str, str] = {}
    data_list: List[str] = []
    header_lines_count = 0
    
    with open(infile) as f:
        lines = [line.rstrip('\n\r') for line in f.readlines()]
    
    reading_data = False
    for i, line in enumerate(lines):
        if not line:  # Skip empty lines
            if not reading_data:
                header_lines_count += 1
            continue
            
        if reading_data:
            data_list.append(line)
        elif line.startswith('#'):
            # Store the next non-empty line as the header value
            key = line[1:]  # Remove the '#'
            for next_line in lines[i+1:]:
                if next_line and not next_line.startswith('#'):
                    headers[key] = next_line
                    break
            header_lines_count += 1
        else:
            header_lines_count += 1
        
        if line == '#GRID':
            reading_data = True
            
    return data_list, headers

def _read_gxf_data(infile: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a GXF file and return the grid data and metadata.
    Following official GXF specifications from Geosoft.
    
    Parameters:
    infile (str): Path to the GXF file
    
    Returns:
    tuple: (grid_array, metadata)
        - grid_array: numpy array containing the properly oriented grid
        - metadata: dictionary containing all GXF parameters
    """
    # First, read header efficiently until #GRID
    headers: Dict[str, str] = {}
    grid_start_line = 0
    
    with open(infile, 'r') as f:
        lines = f.readlines()
    
    # Parse headers until #GRID
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n\r')
        grid_start_line = i + 1
        
        if not line:
            i += 1
            continue
            
        if line.startswith('#'):
            key = line[1:]  # Remove the '#'
            if key == 'GRID':
                break  # Found start of data section
            # Look for header value in next non-empty, non-# line
            j = i + 1
            while j < len(lines):
                next_line = lines[j].rstrip('\n\r')
                if next_line and not next_line.startswith('#'):
                    headers[key] = next_line
                    break
                j += 1
        i += 1
    
    # Parse metadata with proper type conversion
    metadata: Dict[str, Any] = {}
    for key, value in headers.items():
        try:
            # Try converting to float first (handles scientific notation)
            metadata[key] = float(value)
            # If it's a whole number, convert to int
            if metadata[key].is_integer():
                metadata[key] = int(metadata[key])
        except ValueError:
            # If conversion fails, keep as string
            metadata[key] = value.strip()
    
    # Use np.loadtxt efficiently, but handle inconsistent columns by reading as 1D
    try:
        # Try to use np.loadtxt with skiprows for efficiency
        data_1d = np.loadtxt(infile, skiprows=grid_start_line)
        if data_1d.ndim > 1:
            data_1d = data_1d.flatten()
    except ValueError:
        # Fallback: GXF data may have inconsistent columns per line
        # Read the data section manually and flatten
        data_lines = lines[grid_start_line:]
        data_values = []
        for line in data_lines:
            line = line.strip()
            if line:
                values = line.split()
                data_values.extend([float(val) for val in values])
        data_1d = np.array(data_values)
    
    # Get grid dimensions from already-parsed metadata
    nrows = metadata['ROWS']
    ncols = metadata['POINTS']
    
    # Reshape to 2D array
    grid_array = data_1d.reshape((nrows, ncols))
    
    # Handle dummy values
    if 'DUMMY' in metadata:
        dummy_value = metadata['DUMMY']
        grid_array[grid_array == dummy_value] = np.nan
    
    # Handle grid orientation based on SENSE parameter
    sense = str(metadata.get('SENSE', 1))  # Default to 1 if not specified
    if sense == '1':  # First point at bottom left of grid
        grid_array = np.flipud(grid_array)
    
    # Add convenient grid parameters
    metadata.update({
        'nx': ncols,
        'ny': nrows,
        'x_inc': metadata.get('PTSEPARATION', metadata.get('XSEP', 1.0)),
        'y_inc': metadata.get('RWSEPARATION', metadata.get('YSEP', 1.0)),
        'x_min': metadata.get('XORIGIN', 0.0),
        'y_min': metadata.get('YORIGIN', 0.0)
    })
    
    # Add projection information if available
    if 'PRJTYPE' in metadata:
        metadata['projection'] = {
            'type': metadata['PRJTYPE'],
            'units': metadata.get('PRJUNIT', 'unknown'),
            'parameters': {
                'semi_major_axis': metadata.get('A_AXIS_RADIUS'),
                'semi_minor_axis': metadata.get('B_AXIS_RADIUS'),
                'reference_longitude': metadata.get('RFLONGITUDE'),
                'reference_latitude': metadata.get('RFLATITUDE'),
                'first_standard_parallel': metadata.get('FIRST_STANDARD_PARALLEL'),
                'second_standard_parallel': metadata.get('SECOND_STANDARD_PARALLEL'),
                'false_easting': metadata.get('FLSEASTING'),
                'false_northing': metadata.get('FLSNORTHING')
            }
        }
    
    return grid_array, metadata


def read_gxf(infile: str) -> xr.DataArray:
    """
    Read a GXF file and convert it to an xarray DataArray with proper coordinates.
    
    The GXF format is an ASCII file format for gridded data developed by
    Geosoft. This function reads the header information and grid data,
    returning it as an xarray.DataArray for convenient analysis.
    
    Parameters
    ----------
    infile : str
        Path to the GXF file
        
    Returns
    -------
    grid : xarray.DataArray
        An xarray.DataArray containing the grid data with appropriate
        coordinates and metadata from the GXF header stored in attributes.
    """
    # Read the GXF file using existing function
    grid_array, metadata = _read_gxf_data(infile)
    
    # Create coordinate arrays
    x_coords = np.arange(metadata['nx']) * metadata['x_inc'] + metadata['x_min']
    y_coords = np.arange(metadata['ny']) * metadata['y_inc'] + metadata['y_min']
    
    # Determine coordinate names based on rotation
    rotation = float(metadata.get('ROTATION', 0.0))
    if rotation == 0.0:
        x_name, y_name = 'easting', 'northing'
    else:
        x_name, y_name = 'x', 'y'
    
    # Create DataArray with coordinates
    coords = {x_name: x_coords, y_name: y_coords}
    dims = [y_name, x_name]
    
    da = xr.DataArray(
        data=grid_array,
        dims=dims,
        coords=coords,
        name=metadata.get('TITLE', 'GXF_Grid')
    )
    
    # Add all metadata as attributes
    attrs = metadata.copy()
    
    # If projection exists, flatten its nested structure for DataArray attributes
    if 'projection' in attrs:
        proj_info = attrs.pop('projection')
        attrs['projection_type'] = proj_info['type']
        attrs['projection_units'] = proj_info['units']
        # Add non-None projection parameters with a proj_ prefix
        for key, value in proj_info['parameters'].items():
            if value is not None:
                attrs[f'proj_{key}'] = value
    
    da.attrs = attrs
    return da
