"""
Layer of tesseroids
===================
"""
import boule as bl
import ensaio
import numpy as np
import verde as vd
import xarray as xr

import harmonica as hm

fname = ensaio.fetch_earth_topography(version=1)
topo = xr.load_dataarray(fname)

region = (-75, -50, -55, -20)
topo = topo.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))

ellipsoid = bl.WGS84

longitude, latitude = np.meshgrid(topo.longitude, topo.latitude)
reference = ellipsoid.geocentric_radius(latitude)
surface = topo + reference
density = xr.where(topo > 0, 2670.0, 2670.0 - 1040.0)

tesseroids = hm.tesseroid_layer(
    coordinates=(topo.longitude, topo.latitude),
    surface=surface,
    reference=reference,
    properties={"density": density},
)

# Create a regular grid of computation points located at 10km above reference
grid_longitude, grid_latitude = vd.grid_coordinates(region=region, spacing=5)
grid_radius = ellipsoid.geocentric_radius(grid_latitude) + 10e3
grid_coords = (grid_longitude, grid_latitude, grid_radius)

# Compute gravity field of tesseroids on a regular grid of observation points
gravity = tesseroids.tesseroid_layer.gravity(grid_coords, field="g_z")
