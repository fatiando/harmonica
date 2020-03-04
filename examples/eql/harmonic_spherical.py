"""
Gridding and upward continuation in Spherical coordinates
=========================================================

When gridding magnetic or gravity data on large regions thought the equivalent
layer technique, the curvature of the Earth must be taken into account.
Projecting data for large regions may introduce projection errors on the
predictions.

:class:`harmonica.EQLHarmonicSpherical` is able to estimate the
coefficients of a set of point sources (the equivalent layer) that fit the
observed harmonic data given in Spherical coordinates.
It allows to benefit from the same advantages that the equivalent layer
technique offers while taking into account the curvature of the Earth.
"""
import numpy as np
import boule as bl
import verde as vd
import harmonica as hm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# Fetch the sample gravity data from South Africa
data = hm.datasets.fetch_south_africa_gravity()

# Downsample the number of points to speed-up the computations for this example
reducer = vd.BlockMean(spacing=0.2, drop_coords=False)
(longitude, latitude, elevation), gravity_data, _ = reducer.filter(
    (data.longitude, data.latitude, data.elevation), data.gravity,
)

# Compute gravity disturbance by removing the gravity of normal Earth
ellipsoid = bl.WGS84
gamma = ellipsoid.normal_gravity(latitude, height=elevation)
gravity_disturbance = gravity_data - gamma

# Convert data coordinates to spherical
coordinates = ellipsoid.geodetic_to_spherical(longitude, latitude, elevation)

# Create the equivalent layer
eql = hm.EQLHarmonicSpherical(damping=1e-3, relative_depth=10000)

# Fit the layer coefficients to the observed magnetic anomaly
eql.fit(coordinates, gravity_disturbance)

# Evaluate the data fit by calculating an R² score against the observed data.
# This is a measure of how well layer the fits the data NOT how good the
# interpolation will be.
print("R² score:", eql.score(coordinates, gravity_disturbance))

# Interpolate data on a regular grid with 0.5 degrees spacing.
# The interpolation requires an extra coordinate (radius). By passing in
# 1000 m above the maximum observation radius, we're effectively
# upward-continuing the data.
grid_radius = coordinates[2].max() + 1000
grid = eql.grid(
    spacing=0.2,
    extra_coords=grid_radius,
    dims=["spherical_latitude", "longitude"],
    data_names=["gravity_disturbance"],
)

# Mask grid points too far from data points
grid = vd.distance_mask(data_coordinates=coordinates, maxdist=0.5, grid=grid,)

# Plot observed and gridded gravity disturbance
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 9))
ax1.set_aspect("equal")
ax2.set_aspect("equal")

maxabs = vd.maxabs(gravity_disturbance, grid.gravity_disturbance.values)

tmp = ax1.scatter(
    *coordinates[:2],
    c=gravity_disturbance,
    s=3,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax1, label="mGal", pad=0.05, aspect=40, orientation="horizontal")

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax2,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    add_labels=False,
)
plt.colorbar(tmp, ax=ax2, label="mGal", pad=0.05, aspect=40, orientation="horizontal")
ax2.set_xlim(*ax1.get_xlim())

plt.tight_layout()
plt.show()
