"""
Point masses in Cartesian coordinates [all fields]
==================================================

Function :func:`harmonica.point_mass_gravity` can calculate several different fields,
including the gravitational potential and 3-component gravitational acceleration.
This is what they look like.
"""
import harmonica as hm
import verde as vd
import matplotlib.pyplot as plt
import matplotlib.ticker


# Define the coordinates for two point masses
easting = [5e3, 15e3]
northing = [7e3, 13e3]
# The vertical coordinate is positive upward so negative numbers represent depth
upward = [-0.5e3, -1e3]
points = [easting, northing, upward]
# We're using "negative" masses to represent a "mass deficit" since we assume
# measurements are gravity disturbances, not actual gravity values.
masses = [30e11, -100e11]

# Define computation points on a grid at 500m above the ground
coordinates = vd.grid_coordinates(
    region=[0, 20e3, 0, 20e3], shape=(100, 100), extra_coords=500
)

# Compute all available fields
fields = ["g_easting", "g_northing", "g_z", "potential"]
# Store the arrays in a dictionary with the field name as keys
data = {
    field: hm.point_mass_gravity(
        coordinates, points, masses, field=field, coordinate_system="cartesian"
    )
    for field in fields
}
print(data)

# Units to put in the colorbar of each plot
units = {"g_easting": "mGal", "g_northing": "mGal", "g_z": "mGal", "potential": "m²/s²"}

# Plot the results on a map
fig, axes = plt.subplots(2, 3, figsize=(14, 11))
for ax, field in zip(axes.ravel(), fields):
    ax.set_aspect("equal")
    # Get the maximum absolute value so we can center the colorbar on zero
    maxabs = vd.maxabs(data[field])
    img = ax.contourf(
        *coordinates[:2], data[field], 60, vmin=-maxabs, vmax=maxabs, cmap="seismic"
    )
    plt.colorbar(
        img, ax=ax, pad=0.1, aspect=50, label=units[field], orientation="horizontal"
    )
    # Plot the point mass locations
    ax.plot(easting, northing, "oy")
    ax.set_title(field)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
for ax in axes.ravel()[-2:]:
    ax.set_axis_off()
plt.tight_layout()
plt.show()
