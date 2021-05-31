# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Derivatives of a regular grid
=============================
"""
import matplotlib.pyplot as plt
import pyproj
import numpy as np
import verde as vd
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
eql = hm.EQLHarmonic(relative_depth=1000, damping=1).fit(
    coordinates, data.total_field_anomaly_nt
)
grid = eql.grid(upward=1500, spacing=500, data_names=["magnetic_anomaly"])

# The grid is a xarray.Dataset with values, coordinates, and metadata
print("\nGenerated grid:\n", grid)

# Compute the spatial derivatives of the grid along the easting, northing and
# upward directions
deriv_easting = hm.derivative_easting(grid.magnetic_anomaly)
deriv_northing = hm.derivative_northing(grid.magnetic_anomaly)
deriv_upward = hm.derivative_upward(grid.magnetic_anomaly)

# Plot original magnetic anomaly and its derivatives
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2, ncols=2, figsize=(9, 18), sharex=True, sharey=True
)

# Plot the magnetic anomaly grid
grid.magnetic_anomaly.plot(ax=ax1, cbar_kwargs={"label": "nT"})
ax1.set_title("Magnetic anomaly")

# Define keyword arguments for the derivative axes
kwargs = dict(cmap="RdBu_r", cbar_kwargs={"label": "nT/m"})

# Plot the easting derivative
scale = np.quantile(np.abs(deriv_easting), q=0.98)  # scale the colorbar
deriv_easting.plot(ax=ax2, vmin=-scale, vmax=scale, **kwargs)
ax2.set_title("Easting derivative")

# Plot the northing derivative
scale = np.quantile(np.abs(deriv_northing), q=0.98)  # scale the colorbar
deriv_northing.plot(ax=ax3, vmin=-scale, vmax=scale, **kwargs)
ax3.set_title("Northing derivative")

# Plot the upward derivative
scale = np.quantile(np.abs(deriv_upward), q=0.98)  # scale the colorbar
deriv_upward.plot(ax=ax4, vmin=-scale, vmax=scale, **kwargs)
ax4.set_title("Upward derivative")

# Scale the axes
for ax in (ax1, ax2, ax3, ax4):
    ax.set_aspect("equal")

# Set ticklabels with scientific notation
ax1.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

plt.show()
