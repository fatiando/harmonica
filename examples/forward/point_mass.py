"""
Point Masses in Cartesian Coordinates
=====================================

The simplest geometry used to compute gravitational fields are point masses. Although
modelling geologic structures with point masses can be challenging, they are very useful
for other purposes: creating synthetic models, solving inverse problems, generating
equivalent sources for interpolation, etc. The gravitational fields generated by point
masses can be quickly computed either in Cartesian or in geocentric spherical coordinate
systems. We will compute the gravitational acceleration generated by a set of point
masses on a computation grid given in Cartesian coordinates using the
:func:`harmonica.point_mass_gravity` function.
"""
import harmonica as hm
import verde as vd
import matplotlib.pyplot as plt
import matplotlib.ticker


# Define the coordinates for two point masses
easting = [5e3, 15e3]
northing = [5e3, 15e3]
# The vertical coordinate is positive upward so negative numbers represent depth
upward = [-0.5e3, -1e3]
points = [easting, northing, upward]
# We're using "negative" masses to represent a "mass deficit" since we assume
# measurements are gravity disturbances, not actual gravity values.
masses = [3e11, -10e11]

# Define computation points on a grid at 500m above the ground
coordinates = vd.grid_coordinates(
    region=[0, 20e3, 0, 20e3], shape=(80, 80), extra_coords=500
)

# Compute the downward component of the gravitational acceleration
gravity = hm.point_mass_gravity(
    coordinates, points, masses, field="g_z", coordinate_system="cartesian"
)
print(gravity)

# Plot the results on a map
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect("equal")
# Get the maximum absolute value so we can center the colorbar on zero
maxabs = vd.maxabs(gravity)
img = ax.pcolormesh(
    *coordinates[:2], gravity, vmin=-maxabs, vmax=maxabs, cmap="seismic"
)
plt.colorbar(img, ax=ax, pad=0.04, shrink=0.73, label="mGal")
# Plot the point mass locations
ax.plot(easting, northing, "oy")
ax.set_title("Gravitational acceleration (downward)")
# Convert axes units to km
ax.set_xticklabels(ax.get_xticks() * 1e-3)
ax.set_yticklabels(ax.get_yticks() * 1e-3)
ax.set_xlabel("km")
ax.set_ylabel("km")
plt.tight_layout()
plt.show()
