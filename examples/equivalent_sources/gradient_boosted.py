# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gradient-boosted equivalent sources
===================================

When trying to grid a very large dataset, the regular
:class:`harmonica.EquivalentSources` might not be the best option: they will
require a lot of memory for storing the Jacobian matrices involved in the
fitting process of the source coefficients. Instead, we can make use of the
gradient-boosted equivalent sources, introduced in [Soler2021]_ and available
in Harmonica through the :class:`harmonica.EquivalentSourcesGB` class. The
gradient-boosted equivalent sources divide the survey region in overlapping
windows of equal size and fit the source coefficients iteratively, considering
the sources and data points that fall under each window at a time. The order in
which the windows are visited is randomized to improve convergence of the
algorithm.

Here we will produce a grid out of a portion of the ground gravity survey from
South Africa (see :func:`harmonica.datasets.fetch_south_africa_gravity`) using
the gradient-boosted equivalent sources. This particlar dataset is not very
large, in fact we could use the :class:`harmonica.EquivalentSources` instead.
But we will use the :class:`harmonica.EquivalentSourcesGB` for illustrating how
to use them on a small example.

"""
import boule as bl
import matplotlib.pyplot as plt
import pyproj
import verde as vd

import harmonica as hm

# Fetch the sample gravity data from South Africa
data = hm.datasets.fetch_south_africa_gravity()

# Slice a smaller portion of the survey data to speed-up calculations for this
# example
region = [18, 27, -34.5, -27]
inside = vd.inside((data.longitude, data.latitude), region)
data = data[inside]
print("Number of data points:", data.shape[0])
print("Mean height of observations:", data.elevation.mean())

# Since this is a small area, we'll project our data and use Cartesian
# coordinates
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
easting, northing = projection(data.longitude.values, data.latitude.values)
coordinates = (easting, northing, data.elevation)

# Compute the gravity disturbance
ellipsoid = bl.WGS84
data["gravity_disturbance"] = data.gravity - ellipsoid.normal_gravity(
    data.latitude, data.elevation
)

# Create the equivalent sources
# We'll use the block-averaged sources with a block size of 2km and windows of
# 100km x 100km, a damping of 10 and set the sources at a relative depth of
# 9km. By specifying the random_state, we ensure to get the same solution on
# every run.
window_size = 100e3
block_size = 2e3
eqs_gb = hm.EquivalentSourcesGB(
    depth=9e3,
    damping=10,
    window_size=window_size,
    block_size=block_size,
    random_state=42,
)

# Let's estimate the memory required to store the largest Jacobian when using
# these values for the window_size and the block_size.
jacobian_req_memory = eqs_gb.estimate_required_memory(coordinates)
print(f"Required memory for storing the largest Jacobian: {jacobian_req_memory} bytes")

# Fit the sources coefficients to the observed gravity disturbance.
eqs_gb.fit(coordinates, data.gravity_disturbance)

print("Number of sources:", eqs_gb.points_[0].size)

# Evaluate the data fit by calculating an R² score against the observed data.
# This is a measure of how well the sources fit the data, NOT how good the
# interpolation will be.
print("R² score:", eqs_gb.score(coordinates, data.gravity_disturbance))

# Interpolate data on a regular grid with 2 km spacing. The interpolation
# requires the height of the grid points (upward coordinate). By passing in
# 1000 m, we're effectively upward-continuing the data.
grid = eqs_gb.grid(
    upward=1000,
    spacing=2e3,
    data_names="gravity_disturbance",
)
print(grid)

# Plot the original gravity disturbance and the gridded and upward-continued
# version
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), sharey=True)

# Get the maximum absolute value between the original and gridded data so we
# can use the same color scale for both plots and have 0 centered at the white
# color.
maxabs = vd.maxabs(data.gravity_disturbance, grid.gravity_disturbance)

ax1.set_title("Observed gravity disturbance data")
tmp = ax1.scatter(
    easting,
    northing,
    c=data.gravity_disturbance,
    s=5,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax1, label="mGal", pad=0.07, aspect=40, orientation="horizontal")

ax2.set_title("Gridded with gradient-boosted equivalent sources")
tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax2,
    add_colorbar=False,
    add_labels=False,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax2, label="mGal", pad=0.07, aspect=40, orientation="horizontal")

plt.show()
