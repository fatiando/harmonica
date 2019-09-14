"""
Grid total-field magnetic data from Rio de Janeiro
==================================================

Most potential field surveys gather data on several scatter points that might be at
different heights. For a great number of applications we may need to interpolate these
data points into a regular grid. This can be done through an Equivalent Layer
interpolator. We will use :class:`harmonica.EQLHarmonic` to generate a set of point
sources beneath the observation points that predicts the observed data. Then these point
sources will be used to interpolate the data values into a regular grid at a constant
height.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import verde as vd
import harmonica as hm


# Fetch magnetic anomaly data from Rio de Janeiro
data = hm.datasets.fetch_rio_magnetic()

# Reduce region of the survey to speed things up
region = [-42.35, -42.10, -22.35, -22.15]
inside = vd.inside((data.longitude.values, data.latitude.values), region)
data = data[inside]
print("Number of data points:", data.shape[0])

# Project coordinates
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
data["easting"], data["northing"] = projection(
    data.longitude.values, data.latitude.values
)
coordinates = (data.easting, data.northing, data.altitude_m)

train, test = vd.train_test_split(
    coordinates, data.total_field_anomaly_nt, random_state=0
)

eql = hm.EQLHarmonic(depth=1000, damping=10)
eql.fit(*train)
print("R² score on testing set:", eql.score(*test))

# Interpolate data into the regular grid at 200m above the sea level
grid = eql.grid(spacing=400, data_names=["magnetic_anomaly"], extra_coords=200)
# The grid is a xarray.Dataset with values, coordinates, and metadata
print(grid)

# Plot original magnetic anomaly
maxabs = vd.maxabs(data.total_field_anomaly_nt, grid.magnetic_anomaly.values)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
tmp = ax1.scatter(
    data.easting,
    data.northing,
    c=data.total_field_anomaly_nt,
    s=20,
    vmin=-maxabs,
    vmax=maxabs,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax1, label="nT")
ax1.set_title("Observed Anomaly Magnetic data from Rio de Janeiro")

# Plot gridded magnetic anomaly
tmp = grid.magnetic_anomaly.plot.pcolormesh(
    ax=ax2, add_colorbar=False, vmin=-maxabs, vmax=maxabs, cmap="seismic"
)
plt.colorbar(tmp, ax=ax2, label="nT")
ax2.set_title("Gridded Anomaly Magnetic data from Rio de Janeiro")
plt.show()
