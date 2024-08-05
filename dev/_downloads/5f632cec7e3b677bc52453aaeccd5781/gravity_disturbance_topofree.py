# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Topography-free (Bouguer) Gravity Disturbances
==============================================

The gravity disturbance is the observed gravity minus the normal gravity
(:meth:`boule.Ellipsoid.normal_gravity`). The signal that is left is assumed to
be due to the differences between the actual Earth and the reference ellipsoid.
Big components in this signal are the effects of topographic masses outside of
the ellipsoid and residual effects of the oceans (we removed ellipsoid crust
where there was actually ocean water). These two components are relatively well
known and can be removed from the gravity disturbance. The simplest way of
calculating the effects of topography and oceans is through the Bouguer plate
approximation.

We'll use :func:`harmonica.bouguer_correction` to calculate a topography-free
gravity disturbance for Earth using our sample gravity and topography data. One
thing to note is that the ETOPO1 topography is referenced to the geoid, not the
ellipsoid. Since we want to remove the masses between the surface of the Earth
and ellipsoid, we need to add the geoid height to the topography before Bouguer
correction.
"""
import boule as bl
import ensaio
import pygmt
import xarray as xr

import harmonica as hm

# Load the global gravity, topography, and geoid grids
fname_gravity = ensaio.fetch_earth_gravity(version=1)
fname_geoid = ensaio.fetch_earth_geoid(version=1)
fname_topo = ensaio.fetch_earth_topography(version=1)
data = xr.merge(
    [
        xr.load_dataarray(fname_gravity),
        xr.load_dataarray(fname_geoid),
        xr.load_dataarray(fname_topo),
    ]
)
print(data)

# Calculate normal gravity and the disturbance
ellipsoid = bl.WGS84
gamma = ellipsoid.normal_gravity(data.latitude, data.height)
disturbance = data.gravity - gamma

# Reference the topography to the ellipsoid
topography_ell = data.topography + data.geoid

# Calculate the Bouguer planar correction and the topography-free disturbance.
# Use the default densities for the crust and ocean water.
bouguer = hm.bouguer_correction(topography_ell)
disturbance_topofree = disturbance - bouguer

# Make a plot of data using PyGMT
fig = pygmt.Figure()

pygmt.grd2cpt(grid=disturbance_topofree, cmap="vik+h0", continuous=True)

title = "Topography-free (Bouguer) gravity disturbance of the Earth"

with pygmt.config(FONT_TITLE="14p"):
    fig.grdimage(
        region="g",
        projection="G-60/0/15c",
        frame=f"+t{title}",
        grid=disturbance_topofree,
        cmap=True,
    )

fig.coast(shorelines="0.5p,black", resolution="crude")

fig.colorbar(cmap=True, frame=["a200f50", "x+lmGal"])

fig.show()
