# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Airy Isostasy
=============

According to the Airy hypothesis of isostasy, topography above sea level is
supported by a thickening of the crust (a root) while oceanic basins are
supported by a thinning of the crust (an anti-root). Function
:func:`harmonica.isostasy_airy` computes the depth to crust-mantle interface
(the Moho) according to Airy isostasy. One must assume a value for the
reference thickness of the continental crust in order to convert the
root/anti-root thickness into Moho depth. The function contains common default
values for the reference thickness and crust, mantle, and water densities
[TurcotteSchubert2014]_.

We'll use our sample topography data
(:func:`harmonica.datasets.fetch_topography_earth`) to calculate the Airy
isostatic Moho depth of Africa.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the elevation model and cut out the portion of the data corresponding to
# Africa
data = hm.datasets.fetch_topography_earth()
region = (-20, 60, -40, 45)
data_africa = data.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))
print("Topography/bathymetry grid:")
print(data_africa)

# Calculate the isostatic Moho depth using the default values for densities and
# reference Moho
moho = hm.isostasy_airy(data_africa.topography)
print("\nMoho depth grid:")
print(moho)

# Draw the maps
plt.figure(figsize=(8, 9.5))
ax = plt.axes(projection=ccrs.LambertCylindrical(central_longitude=20))
pc = moho.plot.pcolormesh(
    ax=ax, cmap="viridis_r", add_colorbar=False, transform=ccrs.PlateCarree()
)
plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.01, aspect=50, label="meters")
ax.coastlines()
ax.set_title("Airy isostatic Moho depth of Africa")
ax.set_extent(region, crs=ccrs.PlateCarree())
plt.show()
