"""
Grid magnetic data with Harmonic Equivalent Layer interpolator
"""
import harmonica as hm
import verde as vd
import numpy as np
import pyproj
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit


# Fetch magnetic anomaly data from Rio de Janeiro
data = hm.datasets.fetch_rio_magnetic()

# Reduce region to speed things up
region = [-42.9, -42.3, -22.44, -22.14]
are_inside = vd.inside((data.longitude, data.latitude), region)
data = data[are_inside]

# Project coordinates
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
data["easting"], data["northing"] = projection(
    data.longitude.values, data.latitude.values
)

# Decimate data to avoid aliasing
reducer = vd.BlockReduce(reduction=np.median, spacing=1000)
coordinates, (altitude, magnetic_anomaly) = reducer.filter(
    coordinates=(data.easting, data.northing),
    data=(data.altitude_m, data.total_field_anomaly_nt),
)
coordinates = (*coordinates, altitude)

# Perform a cross-validation in order to score the interpolator
gridder = hm.EQLHarmonic()
shuffle = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
scores = vd.cross_val_score(gridder, coordinates, magnetic_anomaly, cv=shuffle)
print("Score: {}".format(np.mean(scores)))

# Interpolate data into the regular grid
grid = gridder.grid(
    region=vd.get_region(coordinates),
    spacing=100,
    data_names=["magnetic_anomaly"],
    extra_coords=coordinates[-1].mean(),
)

# Plot original magnetic anomaly
fig, ax = plt.subplots()
tmp = ax.scatter(data.easting, data.northing, c=data.total_field_anomaly_nt, s=2)
plt.colorbar(tmp, label="nT")
ax.set_aspect("equal")
plt.title("Observed Anomaly Magnetic data from Rio de Janeiro")
plt.show()

# Plot gridded magnetic anomaly
fig, ax = plt.subplots()
tmp = grid.magnetic_anomaly.plot.pcolormesh(ax=ax, add_colorbar=False)
plt.colorbar(tmp, label="nT")
ax.set_aspect("equal")
plt.title("Gridded Anomaly Magnetic data from Rio de Janeiro")
plt.show()
