"""
Gridding in Spherical coordinates
=================================

When gridding magnetic or gravity data on large regions thought the equivalent
layer technique, the curvature of the Earth must be taken into account.
Projecting data for large regions may introduce projection errors on the
predictions.

:class:`harmonica.EQLHarmonicSpherical` is able to estimate the
coefficients of a set of point sources (the equivalent layer) that fit the
observed harmonic data given in Spherical coordinates.
It allows to benefit from the same advantages that the equivalent layer
technique offers while taking into account the curvature of the Earth.
"""
import numpy as np
import boule as bl
import verde as vd
import harmonica as hm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# Fetch the sample gravity data from South Africa
data = hm.datasets.fetch_south_africa_gravity()

# Downsample the number of points to speed-up the computations for this example
reducer = vd.BlockMean(spacing=0.2, drop_coords=False)
(longitude, latitude, elevation), gravity_data, _ = reducer.filter(
    (data.longitude, data.latitude, data.elevation), data.gravity,
)

# Compute gravity disturbance by removing the gravity of normal Earth
ellipsoid = bl.WGS84
gamma = ellipsoid.normal_gravity(latitude, height=elevation)
gravity_disturbance = gravity_data - gamma

# Convert data coordinates to spherical
coordinates = ellipsoid.geodetic_to_spherical(longitude, latitude, elevation)

# Create the equivalent layer
eql = hm.EQLHarmonicSpherical(damping=1e-3, relative_depth=10000)

# Fit the layer coefficients to the observed magnetic anomaly
eql.fit(coordinates, gravity_disturbance)

# Evaluate the data fit by calculating an R² score against the observed data.
# This is a measure of how well layer the fits the data NOT how good the
# interpolation will be.
print("R² score:", eql.score(coordinates, gravity_disturbance))

# Interpolate data on a regular grid with 0.2 degrees spacing defined on
# geodetic coordinates. To do so we need to specify we want grid coordinates to
# be converted to spherical geocentric coordinates before the prediction is
# carried out. This can be done though the projection argument. The
# interpolation requires an extra coordinate (upward height). By passing in
# 2500 m above the maximum observation radius, we're effectively
# upward-continuing the data (maximum height of observation points is 2400 m).
# All the parameters passed to build the grid (region, spacing and
# extra_coords) are in geodetic coordinates.
region = vd.get_region((longitude, latitude))
grid = eql.grid(
    region=region,
    spacing=0.2,
    extra_coords=2500,
    dims=["latitude", "longitude"],
    data_names=["gravity_disturbance"],
    projection=ellipsoid.geodetic_to_spherical,
)

# Mask grid points too far from data points
grid = vd.distance_mask(data_coordinates=coordinates, maxdist=0.5, grid=grid)

# Get the maximum absolute value between the original and gridded data so we
# can use the same color scale for both plots and have 0 centered at the white
# color.
maxabs = vd.maxabs(gravity_disturbance, grid.gravity_disturbance.values)

# Plot observed and gridded gravity disturbance
fig, (ax1, ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 5),
    sharey=True,
    subplot_kw={"projection": ccrs.PlateCarree()},
)
ax1.coastlines()
ax2.coastlines()
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.xlabels_top = False
gl.ylabels_left = False
gl.ylabels_right = False

tmp = ax1.scatter(
    longitude,
    latitude,
    c=gravity_disturbance,
    s=3,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax1, label="mGal", pad=0.07, aspect=40, orientation="horizontal")
ax1.set_extent(region, crs=ccrs.PlateCarree())

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax2,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    add_labels=False,
)
plt.colorbar(tmp, ax=ax2, label="mGal", pad=0.07, aspect=40, orientation="horizontal")
ax2.set_extent(region, crs=ccrs.PlateCarree())

plt.subplots_adjust(wspace=0.05, top=1, bottom=0, left=0.05, right=0.95)
plt.savefig("figure.pdf", dpi=300)
plt.show()
