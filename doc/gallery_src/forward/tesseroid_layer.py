# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Layer of tesseroids
===================
"""

import boule as bl
import ensaio
import numpy as np
import pygmt
import verde as vd
import xarray as xr

import harmonica as hm

fname = ensaio.fetch_earth_topography(version=1)
topo = xr.load_dataarray(fname)

region = (-78, -53, -57, -20)
topo = topo.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))

ellipsoid = bl.WGS84

longitude, latitude = np.meshgrid(topo.longitude, topo.latitude)
reference = ellipsoid.geocentric_radius(latitude)
surface = topo + reference
density = xr.where(topo > 0, 2670.0, 1040.0 - 2670.0)

tesseroids = hm.tesseroid_layer(
    coordinates=(topo.longitude, topo.latitude),
    surface=surface,
    reference=reference,
    properties={"density": density},
)

# Create a regular grid of computation points located at 10km above reference
grid_longitude, grid_latitude = vd.grid_coordinates(region=region, spacing=0.5)
grid_radius = ellipsoid.geocentric_radius(grid_latitude) + 10e3
grid_coords = (grid_longitude, grid_latitude, grid_radius)

# Compute gravity field of tesseroids on a regular grid of observation points
gravity = tesseroids.tesseroid_layer.gravity(grid_coords, field="g_z")
gravity = vd.make_xarray_grid(
    grid_coords,
    gravity,
    data_names="g_z",
    dims=("latitude", "longitude"),
    extra_coords_names="radius",
)

# Plot gravity field
fig = pygmt.Figure()
maxabs = vd.maxabs(gravity.g_z)
pygmt.makecpt(cmap="polar", series=(-maxabs, maxabs))
fig.grdimage(
    gravity.g_z,
    projection="M15c",
    nan_transparent=True,
)
fig.basemap(frame=True)
fig.colorbar(frame='af+l"Gravity [mGal]"', position="JCR")
fig.coast(shorelines="0.5p,black", borders=["1/0.5p,black"])
fig.show()
