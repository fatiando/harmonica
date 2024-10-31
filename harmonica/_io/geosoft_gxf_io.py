# Function to read in GXF files, provided by USGS Generally
# No guarantee this will read in your specific GXF files

# Written by Thomas Martin, Unidata
# Inspired by this gist: https://gist.github.com/jobar8/683483df605a906fb3da747b64627305

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