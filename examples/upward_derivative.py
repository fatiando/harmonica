# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Upward derivative of a regular grid
===================================
"""
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import verde as vd
import xrft

import harmonica as hm

# Fetch the sample total-field magnetic anomaly data from Great Britain
data = hm.datasets.fetch_britain_magnetic()

# Slice a smaller portion of the survey data to speed-up calculations for this
# example
region = [-5.5, -4.7, 57.8, 58.5]
inside = vd.inside((data.longitude, data.latitude), region)
data = data[inside]

# Since this is a small area, we'll project our data and use Cartesian
# coordinates
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
easting, northing = projection(data.longitude.values, data.latitude.values)
coordinates = (easting, northing, data.altitude_m)

# Grid the scatter data using an equivalent layer
eql = hm.EquivalentSources(depth=1000, damping=1).fit(
    coordinates, data.total_field_anomaly_nt
)

grid_coords = vd.grid_coordinates(
    region=vd.get_region(coordinates), spacing=500, extra_coords=1500
)
grid = eql.grid(grid_coords, data_names=["magnetic_anomaly"])
grid = grid.magnetic_anomaly

# Pad the grid to increase accuracy of the FFT filter
pad_width = {
    "easting": grid.easting.size // 3,
    "northing": grid.northing.size // 3,
}
grid = grid.drop_vars("upward")  # drop extra coordinates due to bug in xrft.pad
grid_padded = xrft.pad(grid, pad_width)

# Compute the upward derivative of the grid
deriv_upward = hm.derivative_upward(grid_padded)

# Unpad the derivative grid
deriv_upward = xrft.unpad(deriv_upward, pad_width)

# Show the upward derivative
print("\nUpward derivative:\n", deriv_upward)

# Plot original magnetic anomaly and the upward derivative
fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 8), sharex=True, sharey=True
)

# Plot the magnetic anomaly grid
grid.plot(
    ax=ax1,
    cmap="seismic",
    cbar_kwargs={"label": "nT", "location": "bottom", "shrink": 0.8, "pad": 0.08},
)
ax1.set_title("Magnetic anomaly")

# Plot the upward derivative
scale = np.quantile(np.abs(deriv_upward), q=0.98)  # scale the colorbar
deriv_upward.plot(
    ax=ax2,
    vmin=-scale,
    vmax=scale,
    cmap="seismic",
    cbar_kwargs={"label": "nT/m", "location": "bottom", "shrink": 0.8, "pad": 0.08},
)
ax2.set_title("Upward derivative")

# Scale the axes
for ax in (ax1, ax2):
    ax.set_aspect("equal")

# Set ticklabels with scientific notation
ax1.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

plt.tight_layout()
plt.show()
