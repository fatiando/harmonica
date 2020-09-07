"""
Layer of prisms
===============
"""
import pyproj
import numpy as np
import verde as vd
import harmonica as hm
import rockhound as rh
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# Read ETOPO-1 topography
etopo = rh.fetch_etopo1(version="bedrock")

# Select a subset that corresponds to South Africa
south_africa_topo = etopo.sel(latitude=slice(-35, -18), longitude=slice(12, 33))

# Downsample the topography to speed up computations
south_africa_topo = south_africa_topo.sel(
    longitude=south_africa_topo.longitude[::5], latitude=south_africa_topo.latitude[::5]
)

# Project the grid
projection = pyproj.Proj(proj="merc", lat_ts=south_africa_topo.latitude.values.mean())
south_africa_topo = vd.project_grid(south_africa_topo.bedrock, projection=projection)

# Create layer of prisms
region = (
    south_africa_topo.easting.values.min(),
    south_africa_topo.easting.values.max(),
    south_africa_topo.northing.values.min(),
    south_africa_topo.northing.values.max(),
)
spacing = (
    south_africa_topo.northing.values[1] - south_africa_topo.northing.values[0],
    south_africa_topo.easting.values[1] - south_africa_topo.easting.values[0],
)
prisms = hm.prisms_layer(
    region, spacing=spacing, bottom=None, top=None, properties={"density": 2670}
)
top = south_africa_topo.where(south_africa_topo > 0).fillna(0)
bottom = south_africa_topo.where(south_africa_topo <= 0).fillna(0)
prisms["top"] = top
prisms["bottom"] = bottom

prisms["density"].where(prisms.bottom >= 0, 1000)

prisms.top.plot()
plt.gca().set_aspect("equal")
plt.show()

prisms.bottom.plot()
plt.gca().set_aspect("equal")
plt.show()

# Compute gravity field on a regular grid located at 2000m above the ellipsoid
coordinates = vd.grid_coordinates(
    region=(12, 33, -35, -18), spacing=0.2, extra_coords=2000
)
easting, northing = projection(*coordinates[:2])
coordinates_projected = (easting, northing, coordinates[-1])
prisms_gravity = prisms.prisms_layer.gravity(coordinates_projected, field="g_z")


# Make a plot of the computed gravity
plt.figure(figsize=(7, 6))
ax = plt.axes()
#  ax = plt.axes(projection=ccrs.Mercator())
tmp = ax.pcolormesh(*coordinates[:2], prisms_gravity)
ax.set_title("Terrain gravity effect")
plt.colorbar(tmp, label="mGal")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
