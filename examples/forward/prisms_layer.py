"""
Layer of prisms
===============
"""
import numpy as np
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt


region = (0, 100e3, -40e3, 40e3)
spacing = 2e3
prisms = hm.prisms_layer(
    region=region, spacing=spacing, top=None, bottom=0, properties={"density": 2670}
)
prisms["top"] = 100 * np.exp(
    -((prisms.easting - 50e3) ** 2 + prisms.northing ** 2) / 1e9
)

coordinates = vd.grid_coordinates(region, spacing=spacing, extra_coords=1e3)

gravity = prisms.prisms_layer.gravity(coordinates, field="g_z")

plt.pcolormesh(*coordinates[:2], gravity)
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()
