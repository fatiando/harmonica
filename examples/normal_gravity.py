"""
Normal Gravity
==============

Normal gravity is defined as the magnitude of the gradient of the gravity potential
(gravitational + centrifugal) of a reference ellipsoid. Function
:func:`harmonica.normal_gravity` implements a closed form solution [LiGotze2001]_ to
calculate normal gravity at any latitude and height (it's symmetric in the longitudinal
direction). The ellipsoid in the calculations used can be changed using
:func:`harmonica.set_ellipsoid`.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm
import verde as vd

# Create a global computation grid with 1 degree spacing
region = [0, 360, -90, 90]
longitude, latitude = vd.grid_coordinates(region, spacing=1)

# Compute normal gravity for the WGS84 ellipsoid (the default) on its surface
gamma = hm.normal_gravity(latitude=latitude, height=0)

# Make a plot of the normal gravity using Cartopy
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic())
ax.set_title("Normal gravity of the WGS84 ellipsoid")
ax.coastlines()
pc = ax.pcolormesh(longitude, latitude, gamma, transform=ccrs.PlateCarree())
plt.colorbar(
    pc, label="mGal", orientation="horizontal", aspect=50, pad=0.01, shrink=0.5
)
plt.tight_layout()
plt.show()
