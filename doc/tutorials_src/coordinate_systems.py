"""
.. _coordinate_systems:

Coordinate Systems
==================

Harmonica can handle different bla...
"""
import verde as vd

grid = vd.grid_coordinates(region=(-10, 10, -10, 10), spacing=1)


# %%
# This is a section header
# ------------------------
#
# Bla bla

print(grid[0].size)
