"""
Gravity forward modeling with prisms
====================================

Meh.
"""
import matplotlib.pyplot as plt
import verde as vd
import harmonica as hm

prism = (-500, 500, -1000, 1000, 0, 500)
coordinates = vd.grid_coordinates(
    (-2e3, 2e3, -2e3, 2e3), spacing=10, extra_coords=[-20]
)

gravity = hm.prism_gravity(coordinates, prism, density=2670, field="gz")
print(coordinates, gravity)

plt.figure()
plt.pcolormesh(*coordinates[:2], gravity)
plt.colorbar()
plt.axis("scaled")
plt.show()
