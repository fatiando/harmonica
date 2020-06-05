"""
Gridding in spherical coordinates
=================================

The curvature of the Earth must be taken into account when gridding and
processing magnetic or gravity data on large regions. In these cases,
projecting the data may introduce errors due to the distortions caused by the
projection.

:class:`harmonica.EQLHarmonicSpherical` implements the equivalent layer
technique in spherical coordinates. It has the same advantages as the Cartesian
equivalent layer (:class:`harmonica.EQLHarmonic`) while taking into account the
curvature of the Earth.
"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import boule as bl
import verde as vd
import harmonica as hm


# Fetch the sample gravity data from South Africa
data = hm.datasets.fetch_south_africa_gravity()

# Downsample the data using a blocked mean to speed-up the computations
# for this example. This is preferred over simply discarding points to avoid
# aliasing effects.
blocked_mean = vd.BlockReduce(np.mean, spacing=0.2, drop_coords=False)
(longitude, latitude, elevation), gravity_data = blocked_mean.filter(
    (data.longitude, data.latitude, data.elevation), data.gravity,
)

# Compute gravity disturbance by removing the gravity of normal Earth
ellipsoid = bl.WGS84
gamma = ellipsoid.normal_gravity(latitude, height=elevation)
gravity_disturbance = gravity_data - gamma

# Convert data coordinates from geodetic (longitude, latitude, height) to
# spherical (longitude, spherical_latitude, radius).
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
# geodetic coordinates. By setting this grid at 2500 m above the ellipsoid,
# we're effectively upward-continuing the data (maximum height of observation
# points is 2400 m). Firstly, we must create the regular grid
region = vd.get_region((longitude, latitude))
grid_coords = vd.grid_coordinates(region=region, spacing=0.2, extra_coords=2500)

# Because the gridder is defined in spherical coordinates, we must convert the
# grid coordinates in grodetic to spherical.
grid_coords_spherical = ellipsoid.geodetic_to_spherical(*grid_coords)

# Then we can predict gravity disturbance values on the grid points
grid = eql.predict(grid_coords_spherical)

# Store the resulting grid in a xr.Dataset
dims = ("latitude", "longitude")
coords = {
    "longitude": grid_coords[0][0, :],
    "latitude": grid_coords[1][:, 0],
    "upward": (dims, grid_coords[2]),
}
data_vars = {"gravity_disturbance": (dims, grid)}
grid = xr.Dataset(data_vars, coords=coords)

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
plt.show()
