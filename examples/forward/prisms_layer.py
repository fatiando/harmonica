"""
Layer of prisms
===============
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
