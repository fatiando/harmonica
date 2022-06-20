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
import pygmt

import harmonica as hm

# Load the gravity grid
data = hm.datasets.fetch_gravity_earth()
print(data)

# Make a plot using Pygmt
fig = pygmt.Figure()

title = "Gravity of the Earth (EIGEN-6C4)"

fig.grdimage(
    region="g",
    projection="G150/0/15c",
    frame=f"+t{title}",
    grid=data.gravity,
    cmap="viridis",
)

fig.coast(shorelines="0.5p,black", resolution="crude")

fig.colorbar(cmap=True, frame=["a1000f250", "x+lmGal"])

fig.show()
