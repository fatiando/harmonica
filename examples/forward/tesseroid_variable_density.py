# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Tesseroids with variable density
================================

The :func:`harmonica.tesseroid_gravity` is capable of computing the
gravitational effects of tesseroids whose density is defined through
a continuous function of the radial coordinate. This is achieved by the
application of the method introduced in [Soler2021]_.

To do so we need to define a regular Python function for the density, which
should have a single argument (the ``radius`` coordinate) and return the
density of the tesseroids at that radial coordinate.
In addition, we need to decorate the density function with
:func:`numba.jit(nopython=True)` or ``numba.njit`` for short.

On this example we will show how we can compute the gravitational effect of
four tesseroids whose densities are given by a custom linear ``density``
function.
"""
import boule as bl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import verde as vd
from numba import njit

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

# Define a linear density function. We should use the jit decorator so Numba
# can run the forward model efficiently.


@njit
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
