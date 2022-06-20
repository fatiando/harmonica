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
import boule as bl
import pygmt

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

# Make a plot of data using PyGMT
fig = pygmt.Figure()

pygmt.grd2cpt(grid=disturbance, cmap='polar', continuous=True)

title = "Gravity disturbance of the Earth"

fig.grdimage(
    region='g', 
    projection='G160/0/15c', 
    frame=f'+t{title}',
    grid=disturbance, 
    cmap=True,
    )

fig.coast(shorelines='0.5p,black', resolution='crude')

fig.colorbar(cmap=True, frame=['a100f50', 'x+lmGal'])

fig.show()