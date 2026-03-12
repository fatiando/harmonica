# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Tilt of a regular grid
======================
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

# Compute the tilt of the grid
tilt_grid = hm.tilt_angle(magnetic_grid_padded)

# Unpad the tilt grid
tilt_grid = xrft.unpad(tilt_grid, pad_width)

# Show the tilt
print("\nTilt:\n", tilt_grid)

# Define the inclination and declination of the region by the time of the data
# acquisition (1990).
inclination, declination = -52.98, 6.51

# Apply a reduction to the pole over the magnetic anomaly grid. We will assume
# that the sources share the same inclination and declination as the
# geomagnetic field.
rtp_grid_padded = hm.reduction_to_pole(
    magnetic_grid_padded, inclination=inclination, declination=declination
)

# Unpad the reduced to the pole grid
rtp_grid = xrft.unpad(rtp_grid_padded, pad_width)

# Compute the tilt of the padded rtp grid
tilt_rtp_grid = hm.tilt_angle(rtp_grid_padded)

# Unpad the tilt grid
tilt_rtp_grid = xrft.unpad(tilt_rtp_grid, pad_width)

# Show the tilt from RTP
print("\nTilt from RTP:\n", tilt_rtp_grid)

# Plot original magnetic anomaly, its RTP, and the tilt of both
region = (
    magnetic_grid.easting.values.min(),
    magnetic_grid.easting.values.max(),
    magnetic_grid.northing.values.min(),
    magnetic_grid.northing.values.max(),
)
fig = pygmt.Figure()
with fig.subplot(
    nrows=2,
    ncols=2,
    subsize=("20c", "20c"),
    sharex="b",
    sharey="l",
    margins=["1c", "1c"],
):
    cpt_lim = 0.5 * vd.maxabs(magnetic_grid, rtp_grid)
    with fig.set_panel(panel=0):
        # Make colormap of data
        pygmt.makecpt(cmap="balance+h0", series=[-cpt_lim, cpt_lim], background=True)
        # Plot magnetic anomaly grid
        fig.grdimage(
            grid=magnetic_grid,
            projection="X?",
            cmap=True,
            frame=["a", "+tTotal field anomaly grid"],
        )
    with fig.set_panel(panel=1):
        # Make colormap of data
        pygmt.makecpt(cmap="balance+h0", series=[-cpt_lim, cpt_lim], background=True)
        # Plot reduced to the pole magnetic anomaly grid
        fig.grdimage(
            grid=rtp_grid,
            projection="X?",
            cmap=True,
            frame=["a", "+tReduced to the pole (RTP)"],
        )
        # Add colorbar
        fig.colorbar(
            frame="af+lnT",
            position="JMR+o1/-0.25c+ef",
        )

    cpt_lim = vd.maxabs(tilt_grid, tilt_rtp_grid)
    with fig.set_panel(panel=2):
        # Make colormap for tilt (saturate it a little bit)
        pygmt.makecpt(cmap="balance+h0", series=[-cpt_lim, cpt_lim], background=True)
        # Plot tilt
        fig.grdimage(
            grid=tilt_grid,
            projection="X?",
            cmap=True,
            frame=["a", "+tTilt of total field anomaly grid"],
        )
    with fig.set_panel(panel=3):
        # Make colormap for tilt rtp (saturate it a little bit)
        pygmt.makecpt(cmap="balance+h0", series=[-cpt_lim, cpt_lim], background=True)
        # Plot tilt
        fig.grdimage(
            grid=tilt_rtp_grid,
            projection="X?",
            cmap=True,
            frame=["a", "+tTilt of RTP grid"],
        )
        # Add colorbar
        fig.colorbar(
            frame="af+lradians",
            position="JMR+o1/-0.25c",
        )
fig.show()
