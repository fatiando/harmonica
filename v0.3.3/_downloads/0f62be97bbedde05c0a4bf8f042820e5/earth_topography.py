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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the topography grid
data = hm.datasets.fetch_topography_earth()
print(data)

# Make a plot of data using Cartopy
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-30))
pc = data.topography.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap="terrain"
)
plt.colorbar(
    pc, label="meters", orientation="horizontal", aspect=50, pad=0.01, shrink=0.6
)
ax.set_title("Topography of the Earth (ETOPO1)")
ax.coastlines()
plt.show()
