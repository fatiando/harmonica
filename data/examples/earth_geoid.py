# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Earth Geoid
===========

The geoid is the equipotential surface of the Earth's gravity potential that
coincides with mean sea level. It's often represented by "geoid heights", which
indicate the height of the geoid relative to the reference ellipsoid (WGS84 in
this case). Negative values indicate that the geoid is below the ellipsoid
surface and positive values that it is above. The data are on a regular grid
with 0.5 degree spacing and was generated from the spherical harmonic model
EIGEN-6C4 [Forste_etal2014]_.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the geoid grid
data = hm.datasets.fetch_geoid_earth()
print(data)

# Make a plot of data using Cartopy
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=100))
pc = data.geoid.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)
plt.colorbar(
    pc, label="meters", orientation="horizontal", aspect=50, pad=0.01, shrink=0.6
)
ax.set_title("Geoid heights (EIGEN-6C4)")
ax.coastlines()
plt.show()
