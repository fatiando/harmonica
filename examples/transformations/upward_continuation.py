# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Upward continuation of a regular grid
=====================================
"""
import ensaio
import pygmt
import verde as vd
import xarray as xr
import xrft

import harmonica as hm

# Fetch magnetic grid over the Lightning Creek Sill Complex, Australia using
# Ensaio and load it with Xarray
fname = ensaio.fetch_lightning_creek_magnetic(version=1)
magnetic_grid = xr.load_dataarray(fname)

# Pad the grid to increase accuracy of the FFT filter
pad_width = {
    "easting": magnetic_grid.easting.size // 3,
    "northing": magnetic_grid.northing.size // 3,
}
magnetic_grid_no_height = magnetic_grid.drop_vars(
    "height"
)  # drop the extra height coordinate
magnetic_grid_padded = xrft.pad(magnetic_grid_no_height, pad_width)

# Upward continue the magnetic grid, from 500 m to 1000 m
# (a height displacement of 500m)
upward_continued = hm.upward_continuation(magnetic_grid_padded, height_displacement=500)

# Unpad the upward continued grid
upward_continued = xrft.unpad(upward_continued, pad_width)

# Show the upward continued grid
print("\nUpward continued magnetic grid:\n", upward_continued)


# Plot original magnetic anomaly and the upward continued grid
fig = pygmt.Figure()
with fig.subplot(nrows=1, ncols=2, figsize=("28c", "15c"), sharey="l"):
    # Make colormap for both plots data
    scale = 2500
    pygmt.makecpt(cmap="polar+h", series=[-scale, scale], background=True)
    with fig.set_panel(panel=0):
        # Plot magnetic anomaly grid
        fig.grdimage(
            grid=magnetic_grid,
            projection="X?",
            cmap=True,
        )
        # Add colorbar
        fig.colorbar(
            frame='af+l"Magnetic anomaly at 500m [nT]"',
            position="JBC+h+o0/1c+e",
        )
    with fig.set_panel(panel=1):
        # Plot upward continued grid
        fig.grdimage(grid=upward_continued, projection="X?", cmap=True)
        # Add colorbar
        fig.colorbar(
            frame='af+l"Upward continued to 1000m [nT]"',
            position="JBC+h+o0/1c+e",
        )
fig.show()
