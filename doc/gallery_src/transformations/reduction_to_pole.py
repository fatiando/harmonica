# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Reduction to the pole of a magnetic anomaly grid
================================================
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
# drop the extra height coordinate
magnetic_grid_no_height = magnetic_grid.drop_vars("height")
magnetic_grid_padded = xrft.pad(magnetic_grid_no_height, pad_width)

# Define the inclination and declination of the region by the time of the data
# acquisition (1990).
inclination, declination = -52.98, 6.51

# Apply a reduction to the pole over the magnetic anomaly grid. We will assume
# that the sources share the same inclination and declination as the
# geomagnetic field.
rtp_grid = hm.reduction_to_pole(
    magnetic_grid_padded, inclination=inclination, declination=declination
)

# Unpad the reduced to the pole grid
rtp_grid = xrft.unpad(rtp_grid, pad_width)

# Show the reduced to the pole grid
print("\nReduced to the pole magnetic grid:\n", rtp_grid)


# Plot original magnetic anomaly and the reduced to the pole
fig = pygmt.Figure()
with fig.subplot(nrows=1, ncols=2, figsize=("28c", "15c"), sharey="l"):
    # Make colormap for both plots
    cpt_lim = 0.5 * vd.maxabs(magnetic_grid, rtp_grid)
    pygmt.makecpt(cmap="balance+h0", series=[-cpt_lim, cpt_lim], background=True)
    with fig.set_panel(panel=0):
        # Plot magnetic anomaly grid
        fig.grdimage(
            grid=magnetic_grid,
            projection="X?",
            cmap=True,
        )
        # Add colorbar
        fig.colorbar(
            frame='af+l"Magnetic anomaly [nT]"',
            position="JBC+h+o0/1c+ef",
        )
    with fig.set_panel(panel=1):
        # Plot upward reduced to the pole grid
        fig.grdimage(grid=rtp_grid, projection="X?", cmap=True)
        # Add colorbar
        fig.colorbar(
            frame='af+l"Reduced to the pole [nT]"',
            position="JBC+h+o0/1c+ef",
        )
fig.show()
