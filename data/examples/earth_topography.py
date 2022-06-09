# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Earth Topography
================

The topography and bathymetry of the Earth according to the ETOPO1 model
[AmanteEakins2009]_. The original model has 1 arc-minute grid spacing but here
we downsampled to 0.5 degree grid spacing to save space and download times.
Heights are referenced to sea level.
"""
import pygmt
import harmonica as hm

# Load the topography grid
data = hm.datasets.fetch_topography_earth()
print(data)

# Make a plot of data using PyGMT
fig = pygmt.Figure()

title = "Topography of the Earth (ETOPO1)"

fig.grdimage(
    region='g',
    projection='G-30/0/15c',
    frame=f'+t{title}',
    grid=data.topography, 
    cmap='globe')

fig.coast(shorelines='0.5p,black', resolution='crude')

fig.colorbar(cmap=True, frame=['a2000f500', 'x+lmeters'])

fig.show()