# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Airy Isostatic Moho
===================

According to the Airy hypothesis of isostasy, topography above sea level is
supported by a thickening of the crust (a root) while oceanic basins are
supported by a thinning of the crust (an anti-root). Function
:func:`harmonica.isostatic_moho_airy` computes the depth to crust-mantle
interface (the Moho) according to Airy isostasy. The function takes the depth
to the crystalline basement and optionally any layers on top of it. Each layer
is defined by its thickness and its density. In addition, one must assume
a value for the reference thickness of the continental crust in order to
convert the root/anti-root thickness into Moho depth. The function contains
common default values for the reference thickness and crust, mantle
[TurcotteSchubert2014]_.

We'll use our sample topography data
(:func:`harmonica.datasets.fetch_topography_earth`) to calculate the Airy
isostatic Moho depth of Africa.
"""
import numpy as np
import pygmt

import harmonica as hm

# Load the elevation model and cut out the portion of the data corresponding to
# Africa
data = hm.datasets.fetch_topography_earth()
region = (-20, 60, -40, 45)
data_africa = data.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))
print("Topography/bathymetry grid:")
print(data_africa)

# Calculate the water thickness
oceans = np.array(data_africa.topography < 0)
water_thickness = data_africa.topography * oceans * -1
water_density = 1030

# Calculate the isostatic Moho depth using the default values for densities and
# reference Moho with water load. We neglect the effect of sediment here, so
# basement elevation refers to topography.
moho = hm.isostatic_moho_airy(
    basement=data_africa.topography,
    layers={"water": (water_thickness, water_density)},
)
print("\nMoho depth grid:")
print(moho)

# Draw the maps
fig = pygmt.Figure()

pygmt.grd2cpt(grid=moho, cmap="viridis", reverse=True, continuous=True)

title = "Airy isostatic Moho depth of Africa"

fig.grdimage(
    region=region,
    projection="Y20/0/10c",
    frame=["ag", f"+t{title}"],
    grid=moho,
    cmap=True,
)

fig.coast(shorelines="0.5p,black", resolution="crude")

fig.colorbar(cmap=True, frame=["a10000f2500", "x+lmeters"])

fig.show()
