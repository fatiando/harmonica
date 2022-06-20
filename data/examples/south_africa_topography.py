# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
South Africa Topography
=======================

The topography and bathymetry of South Africa according to the ETOPO1 model
[AmanteEakins2009]_. The original model has 1 arc-minute grid spacing but here
we downsampled to 0.1 degree grid spacing to save space and download times.
Heights are referenced to sea level.
"""
import pygmt
import verde as vd
import harmonica as hm

# Load the topography grid
data = hm.datasets.fetch_south_africa_topography()
print(data)

# Get the region of the grid
region = vd.get_region((data.longitude.values, data.latitude.values))

# Make a plot using PyGMT
fig = pygmt.Figure()

title = "Topography of South africa (ETOPO1)"   

fig.grdimage(
    region=region,
    projection='M15c',
    grid=data.topography, 
    frame=['ag', f'+t{title}'], 
    cmap='earth',
    )

fig.colorbar(cmap=True, frame=['a2000f500', 'x+lmeters'])

fig.coast(shorelines='1p,black')

fig.show()