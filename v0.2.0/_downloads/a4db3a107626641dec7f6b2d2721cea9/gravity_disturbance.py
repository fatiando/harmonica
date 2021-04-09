# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gravity Disturbances
====================

Gravity disturbances are the differences between the measured gravity and
a reference (normal) gravity produced by an ellipsoid. The disturbances are
what allows geoscientists to infer the internal structure of the Earth. We'll
use the :meth:`boule.Ellipsoid.normal_gravity` function from :mod:`boule` to
calculate the global gravity disturbance of the Earth using our sample gravity
data.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import boule as bl
import harmonica as hm

# Load the global gravity grid
data = hm.datasets.fetch_gravity_earth()
print(data)

# Calculate normal gravity using the WGS84 ellipsoid
ellipsoid = bl.WGS84
gamma = ellipsoid.normal_gravity(data.latitude, data.height_over_ell)
# The disturbance is the observed minus normal gravity (calculated at the
# observation point)
disturbance = data.gravity - gamma

# Make a plot of data using Cartopy
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=160))
pc = disturbance.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap="seismic"
)
plt.colorbar(
    pc, label="mGal", orientation="horizontal", aspect=50, pad=0.01, shrink=0.5
)
ax.set_title("Gravity of disturbance of the Earth")
ax.coastlines()
plt.show()
