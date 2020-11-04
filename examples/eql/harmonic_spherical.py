# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gridding in spherical coordinates
=================================

The curvature of the Earth must be taken into account when gridding and
processing magnetic or gravity data on large regions. In these cases,
projecting the data may introduce errors due to the distortions caused by the
projection.

:class:`harmonica.EQLHarmonicSpherical` implements the equivalent layer
technique in spherical coordinates. It has the same advantages as the Cartesian
equivalent layer (:class:`harmonica.EQLHarmonic`) while taking into account the
curvature of the Earth.
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import boule as bl
import verde as vd
import harmonica as hm


# Fetch the sample gravity data from South Africa
data = hm.datasets.fetch_south_africa_gravity()

# Downsample the data using a blocked mean to speed-up the computations
# for this example. This is preferred over simply discarding points to avoid
# aliasing effects.
blocked_mean = vd.BlockReduce(np.mean, spacing=0.2, drop_coords=False)
(longitude, latitude, elevation), gravity_data = blocked_mean.filter(
    (data.longitude, data.latitude, data.elevation),
    data.gravity,
)

# Compute gravity disturbance by removing the gravity of normal Earth
ellipsoid = bl.WGS84
gamma = ellipsoid.normal_gravity(latitude, height=elevation)
gravity_disturbance = gravity_data - gamma

# Convert data coordinates from geodetic (longitude, latitude, height) to
# spherical (longitude, spherical_latitude, radius).
coordinates = ellipsoid.geodetic_to_spherical(longitude, latitude, elevation)

# Create the equivalent layer
eql = hm.EQLHarmonicSpherical(damping=1e-3, relative_depth=10000)

# Fit the layer coefficients to the observed magnetic anomaly
eql.fit(coordinates, gravity_disturbance)

# Evaluate the data fit by calculating an R² score against the observed data.
# This is a measure of how well layer the fits the data NOT how good the
# interpolation will be.
print("R² score:", eql.score(coordinates, gravity_disturbance))

# Interpolate data on a regular grid with 0.2 degrees spacing. The
# interpolation requires the radius of the grid points (upward coordinate). By
# passing in the maximum radius of the data, we're effectively
# upward-continuing the data. The grid will be defined in spherical
# coordinates.
grid = eql.grid(
    upward=coordinates[-1].max(),
    spacing=0.2,
    data_names=["gravity_disturbance"],
)

# The grid is a xarray.Dataset with values, coordinates, and metadata
print("\nGenerated grid:\n", grid)

# Mask grid points too far from data points
grid = vd.distance_mask(data_coordinates=coordinates, maxdist=0.5, grid=grid)

# Get the maximum absolute value between the original and gridded data so we
# can use the same color scale for both plots and have 0 centered at the white
# color.
maxabs = vd.maxabs(gravity_disturbance, grid.gravity_disturbance.values)

# Get the region boundaries
region = vd.get_region(coordinates)

# Plot observed and gridded gravity disturbance
fig, (ax1, ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 5),
    sharey=True,
)

tmp = ax1.scatter(
    longitude,
    latitude,
    c=gravity_disturbance,
    s=3,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax1, label="mGal", pad=0.07, aspect=40, orientation="horizontal")
ax1.set_aspect("equal")
ax1.set_xlim(*region[:2])
ax1.set_ylim(*region[2:])

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax2,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    add_labels=False,
)
plt.colorbar(tmp, ax=ax2, label="mGal", pad=0.07, aspect=40, orientation="horizontal")
ax2.set_aspect("equal")
ax2.set_xlim(*region[:2])
ax2.set_ylim(*region[2:])

plt.subplots_adjust(wspace=0.05, top=1, bottom=0, left=0.05, right=0.95)
plt.show()
