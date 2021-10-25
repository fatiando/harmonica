# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Tesseroids with variable density
================================
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd
import boule as bl
import harmonica as hm


# Use the WGS84 ellipsoid to obtain the mean Earth radius which we'll use to
# reference the tesseroid
ellipsoid = bl.WGS84
mean_radius = ellipsoid.mean_radius

# Define tesseroid with top surface at the mean Earth radius, a thickness of
# 10km and a linear density function
tesseroids = (
    [-70, -60, -40, -30, mean_radius - 3e3, mean_radius],
    [-70, -60, -30, -20, mean_radius - 5e3, mean_radius],
    [-60, -50, -40, -30, mean_radius - 7e3, mean_radius],
    [-60, -50, -30, -20, mean_radius - 10e3, mean_radius],
)


def density(radius):
    """Linear density function"""
    top = mean_radius
    bottom = mean_radius - 10e3
    density_top = 2670
    density_bottom = 3000
    slope = (density_top - density_bottom) / (top - bottom)
    return slope * (radius - bottom) + density_bottom


# Define computation points on a regular grid at 100km above the mean Earth
# radius
coordinates = vd.grid_coordinates(
    region=[-80, -40, -50, -10],
    shape=(80, 80),
    extra_coords=100e3 + ellipsoid.mean_radius,
)

# Compute the radial component of the acceleration
gravity = hm.tesseroid_gravity(coordinates, tesseroids, density, field="g_z")
print(gravity)

# Plot the gravitational field
fig = plt.figure(figsize=(8, 9))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-60))
img = ax.pcolormesh(*coordinates[:2], gravity, transform=ccrs.PlateCarree())
plt.colorbar(img, ax=ax, pad=0, aspect=50, orientation="horizontal", label="mGal")
ax.coastlines()
ax.set_title("Downward component of gravitational acceleration")
plt.show()