# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Layer of prisms
===============

One way to model three dimensional structures is to create a set of prisms that
approximates their geometry and its physical properties (density,
susceptibility, etc.). The :func:`harmonica.prism_layer` offers a simple way
to create a layer of prisms: a regular grid of prisms of equal size on the
horizontal directions with variable top and bottom boundaries. It returns
a :class:`xarray.Dataset` with the coordinates of the centers of the prisms and
their corresponding physical properties.
The :class:`harmonica.DatasetAccessorPrismsLayer` Dataset accessor can be used
to obtain some properties of the layer like its shape and size or the
boundaries of any prism in the layer. The methods of this Dataset accessor can
be used together with the :func:`harmonica.prism_gravity` to compute the
gravitational effect of the layer.
"""
import numpy as np
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt


# Create a layer of prisms
region = (0, 100e3, -40e3, 40e3)
spacing = 2e3
(easting, northing) = vd.grid_coordinates(region=region, spacing=spacing)
surface = 100 * np.exp(-((easting - 50e3) ** 2 + northing ** 2) / 1e9)
density = 2670.0 * np.ones_like(surface)
prisms = hm.prism_layer(
    coordinates=(easting[0, :], northing[:, 0]),
    surface=surface,
    reference=0,
    properties={"density": density},
)

# Compute gravity field of prisms on a regular grid of observation points
coordinates = vd.grid_coordinates(region, spacing=spacing, extra_coords=1e3)
gravity = prisms.prism_layer.gravity(coordinates, field="g_z")

# Plot gravity field
plt.pcolormesh(*coordinates[:2], gravity)
plt.colorbar(label="mGal", shrink=0.8)
plt.gca().set_aspect("equal")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
plt.title("Gravity acceleration of a layer of prisms")
plt.xlabel("easting [m]")
plt.ylabel("northing [m]")
plt.tight_layout()
plt.show()
