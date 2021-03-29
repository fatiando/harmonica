# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Earth Gravity
=============

This is the magnitude of the gravity vector of the Earth (gravitational
+ centrifugal) at 10 km height. The data is on a regular grid with 0.5 degree
spacing at 10km ellipsoidal height. It was generated from the spherical
harmonic model EIGEN-6C4 [Forste_etal2014]_.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the gravity grid
data = hm.datasets.fetch_gravity_earth()
print(data)

# Make a plot of data using Cartopy
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=150))
pc = data.gravity.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False
)
plt.colorbar(
    pc, label="mGal", orientation="horizontal", aspect=50, pad=0.01, shrink=0.6
)
ax.set_title("Gravity of the Earth (EIGEN-6C4)")
ax.coastlines()
plt.show()
