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
import pygmt 
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
xy_region = vd.get_region(coordinates)
# Grid the scatter data using an equivalent layer
eql = hm.EquivalentSources(depth=1000, damping=1).fit(
    coordinates, data.total_field_anomaly_nt
)

grid_coords = vd.grid_coordinates(
    region=xy_region, spacing=500, extra_coords=1500
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
fig = pygmt.Figure()

# Set figure properties
w, e, s, n = xy_region
fig_height = 10
fig_width = fig_height * (e - w) / (n - s)
fig_ratio = (n - s) / (fig_height / 100)
fig_proj = f"x1:{fig_ratio}"

# Plot the magnetic anomaly grid
title = "Magnetic anomaly"

# Make colormap of data
pygmt.makecpt(
    cmap="vik",
    series=(-(np.quantile(grid, q=1)), np.quantile(grid, q=1)),
    background=True,)

with pygmt.config(FONT_TITLE="16p"):
    fig.grdimage(
        region=xy_region,
        projection=fig_proj,
        grid=grid,
        frame=[f"WSne+t{title}", "xa10000+a15+leasting", "y+lnorthing"],
        cmap=True,)

fig.colorbar(cmap=True, frame=["a200f50", "x+lnT"])

fig.shift_origin(xshift=fig_width + 1)

# Plot the upward derivative
title = "Upward derivative"

# Make colormap of data
pygmt.makecpt(
    cmap="vik",
    series=(-(np.quantile(deriv_upward, q=0.99)), np.quantile(deriv_upward, q=0.99)),
    background=True,)

with pygmt.config(FONT_TITLE="16p"):
    fig.grdimage(
        region=xy_region,
        projection=fig_proj,
        grid=deriv_upward,
        frame=[f"ESnw+t{title}", "xa10000+a15+leasting", "y+lnorthing"],
        cmap=True,)

fig.colorbar(cmap=True, frame=["a.05f.025", "x+lnT/m"])

fig.show()