"""
Layer of prisms
===============

One way to model three dimensional structures is to create a set of prisms that
approximates their geometry and its physical properties (density,
susceptibility, etc.). The :func:`harmonica.prisms_layer` offers a simple way
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
prisms = hm.prisms_layer(
    coordinates=(easting[0, :], northing[:, 0]),
    surface=surface,
    reference=0,
    properties={"density": density},
)

# Compute gravity field of prisms on a regular grid of observation points
coordinates = vd.grid_coordinates(region, spacing=spacing, extra_coords=1e3)
gravity = hm.prism_gravity(
    coordinates,
    prisms.prisms_layer.get_prisms(),
    density=prisms.density.values,
    field="g_z",
)

# Plot gravity field
plt.pcolormesh(*coordinates[:2], gravity)
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()
