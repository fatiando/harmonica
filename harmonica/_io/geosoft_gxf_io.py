# Function to read in GXF files, provided by USGS Generally
# No guarantee this will read in your specific GXF files

# Written by Thomas Martin, Unidata
# Inspired by this gist: https://gist.github.com/jobar8/683483df605a906fb3da747b64627305

# Information about the GXF format found on a USGS Readme:

"""
1. Grid eXchange Format (*.gxf)

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
    # Read entire file
    with open(infile) as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Create dictionary with headers and parameters
    headers: Dict[str, str] = {}
    data_list: List[str] = []
    reading_data = False
    
    for i, line in enumerate(lines):
        if not line:  # Skip empty lines
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
        
        if line == '#GRID':
            reading_data = True
            
    return data_list, headers

def read_gxf(infile: str) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    # Get raw data and headers
    data_list, headers = read_gxf_raw(infile)
    
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
    
    # Convert data strings to numpy array
    data_1d = np.array([])
    for line in data_list:
        # Handle both space and tab-delimited data
        values = np.fromstring(line, sep=' ')
        data_1d = np.concatenate((data_1d, values))
    
    # Get grid dimensions
    nrows = int(headers['ROWS'])
    ncols = int(headers['POINTS'])
    
    # Reshape to 2D array
    grid_array = data_1d.reshape((nrows, ncols))
    
    # Handle dummy values
    dummy_value = float(headers['DUMMY'])
    grid_array[grid_array == dummy_value] = np.nan
    
    # Handle grid orientation based on SENSE parameter
    sense = str(headers.get('SENSE', '1'))  # Default to '1' if not specified
    if sense == '1':  # First point at bottom left of grid
        grid_array = np.flipud(grid_array)
    
    # Add convenient grid parameters
    metadata.update({
        'nx': ncols,
        'ny': nrows,
        'x_inc': float(headers.get('PTSEPARATION', headers.get('XSEP', 1.0))),
        'y_inc': float(headers.get('RWSEPARATION', headers.get('YSEP', 1.0))),
        'x_min': float(headers.get('XORIGIN', 0.0)),
        'y_min': float(headers.get('YORIGIN', 0.0))
    })
    
    # Add projection information if available
    if 'PRJTYPE' in headers:
        metadata['projection'] = {
            'type': headers['PRJTYPE'],
            'units': headers.get('PRJUNIT', 'unknown'),
            'parameters': {
                'semi_major_axis': float(headers['A_AXIS_RADIUS']) if 'A_AXIS_RADIUS' in headers else None,
                'semi_minor_axis': float(headers['B_AXIS_RADIUS']) if 'B_AXIS_RADIUS' in headers else None,
                'reference_longitude': float(headers['RFLONGITUDE']) if 'RFLONGITUDE' in headers else None,
                'reference_latitude': float(headers['RFLATITUDE']) if 'RFLATITUDE' in headers else None,
                'first_standard_parallel': float(headers['FIRST_STANDARD_PARALLEL']) if 'FIRST_STANDARD_PARALLEL' in headers else None,
                'second_standard_parallel': float(headers['SECOND_STANDARD_PARALLEL']) if 'SECOND_STANDARD_PARALLEL' in headers else None,
                'false_easting': float(headers['FLSEASTING']) if 'FLSEASTING' in headers else None,
                'false_northing': float(headers['FLSNORTHING']) if 'FLSNORTHING' in headers else None
            }
        }
    
    return grid_array, metadata

def get_grid_info(metadata: Dict[str, Any]) -> None:
    """
    Print comprehensive information about the GXF grid.
    
    Parameters:
    metadata (dict): Metadata dictionary from read_gxf
    """
    print("=== Grid Information ===")
    print(f"Title: {metadata.get('TITLE', 'Not specified')}")
    print(f"Grid size: {metadata['nx']} x {metadata['ny']} points")
    print(f"Cell size: {metadata['x_inc']} x {metadata['y_inc']}")
    print(f"Origin: ({metadata['x_min']}, {metadata['y_min']})")
    print(f"Rotation: {metadata.get('ROTATION', 0)}")
    print(f"Dummy value: {metadata.get('DUMMY', 'Not specified')}")
    
    if 'projection' in metadata:
        print("\n=== Projection Information ===")
        print(f"Type: {metadata['projection']['type']}")
        print(f"Units: {metadata['projection']['units']}")
        
        params = metadata['projection']['parameters']
        if any(params.values()):
            print("\nProjection Parameters:")
            for key, value in params.items():
                if value is not None:
                    print(f"{key}: {value}")

def gxf_to_xarray(infile: str) -> xr.DataArray:
    """
    Read a GXF file and convert it to an xarray DataArray with proper coordinates.
    
    Parameters:
    infile (str): Path to the GXF file
    
    Returns:
    xarray.DataArray: Georeferenced data array with coordinates and attributes
    """
    # Read the GXF file
    grid_array, metadata = read_gxf(infile)
    
    # Create coordinate arrays
    x_coords = np.arange(metadata['nx']) * metadata['x_inc'] + metadata['x_min']
    y_coords = np.arange(metadata['ny']) * metadata['y_inc'] + metadata['y_min']
    
    # Create DataArray with coordinates
    da = xr.DataArray(
        data=grid_array,
        dims=['y', 'x'],
        coords={
            'x': x_coords,
            'y': y_coords
        },
        name=metadata.get('TITLE', 'GXF_Grid')
    )
    
    # Add all metadata as attributes
    attrs = metadata.copy()
    
    # If projection exists, flatten its nested structure
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
