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

import ensaio
import pygmt
import verde as vd
import xarray as xr

import harmonica as hm

# Fetch magnetic grid over the Lightning Creek Sill Complex, Australia using
# Ensaio and load it with Xarray
fname = ensaio.fetch_lightning_creek_magnetic(version=1)
magnetic_grid = xr.load_dataarray(fname)

# Compute the upward derivative of the grid
deriv_upward = hm.derivative_upward(magnetic_grid)
# Show the upward derivative
print("\nUpward derivative:\n", deriv_upward)


# Plot original magnetic anomaly and the upward derivative
fig = pygmt.Figure()
with fig.subplot(nrows=1, ncols=2, figsize=("28c", "15c"), sharey="l"):
    with fig.set_panel(panel=0):
        # Make colormap of data
        maxabs = vd.maxabs(magnetic_grid, percentile=99)
        pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
        # Plot magnetic anomaly grid
        fig.grdimage(
            grid=magnetic_grid,
            projection="X?",
            cmap=True,
            frame="+tMagnetic Anomaly",
        )
        # Add colorbar
        fig.colorbar(
            frame="af+lnT",
            position="JBC+h+o0/1c+e",
        )
    with fig.set_panel(panel=1):
        # Make colormap for upward derivative
        maxabs = vd.maxabs(deriv_upward, percentile=99)
        pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
        # Plot upward derivative
        fig.grdimage(
            grid=deriv_upward,
            projection="X?",
            cmap=True,
            frame="+tUpward Derivative",
        )
        # Add colorbar
        fig.colorbar(
            frame="af+lnT/m",
            position="JBC+h+o0/1c+e",
        )
fig.show()
