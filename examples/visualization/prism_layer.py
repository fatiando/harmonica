# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
3D plot of layer of prisms
==========================

"""
import pyproj
import pyvista as pv
import verde as vd

import harmonica as hm

# Read South Africa topography
south_africa_topo = hm.datasets.fetch_south_africa_topography()

# Project the grid
projection = pyproj.Proj(proj="merc", lat_ts=south_africa_topo.latitude.values.mean())
south_africa_topo = vd.project_grid(south_africa_topo.topography, projection=projection)

# Create a 2d array with the density of the prisms Points above the geoid will
# have a density of 2670 kg/m^3 Points below the geoid will have a density
# contrast equal to the difference between the density of the ocean and the
# density of the upper crust: # 1000 kg/m^3 - 2900 kg/m^3
density = south_africa_topo.copy()  # copy topography to a new xr.DataArray
density.values[:] = 2670.0  # replace every value for the density of the topography
# Change density values of ocean points
density = density.where(south_africa_topo >= 0, 1000 - 2900)

# Create layer of prisms
prisms = hm.prism_layer(
    (south_africa_topo.easting, south_africa_topo.northing),
    surface=south_africa_topo,
    reference=0,
    properties={"density": density},
)

# Create a pyvista UnstructuredGrid from the prism layer
pv_grid = prisms.prism_layer.to_pyvista()
print(pv_grid)

# Plot with pyvista
plotter = pv.Plotter()
plotter.add_mesh(pv_grid, scalars="density")
plotter.set_scale(zscale=75)  # exaggerate the vertical coordinate
plotter.camera_position = "xz"
plotter.camera.elevation = 15
plotter.show()
