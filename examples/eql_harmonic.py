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

# Reduce number of data points to speed things up
data = data.sample(3000, random_state=1)

# Project coordinates
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
data["easting"], data["northing"] = projection(
    data.longitude.values, data.latitude.values
)
coordinates = (data["easting"], data["northing"], data.altitude_m)

# Perform a cross-validation in order to score the interpolator
gridder = hm.EQLHarmonic()
gridder.set_params(depth_factor=1)
shuffle = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1)
scores = vd.cross_val_score(
    gridder, coordinates, data.total_field_anomaly_nt, cv=shuffle
)
print("Score: {}".format(np.mean(scores)))

# Interpolate data into the regular grid at 200m above the sea level
grid = gridder.grid(
    region=vd.get_region(coordinates),
    spacing=100,
    data_names=["magnetic_anomaly"],
    extra_coords=200,
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
tmp = grid.magnetic_anomaly.plot.pcolormesh(ax=ax, add_colorbar=False, cmap="viridis")
plt.colorbar(tmp, label="nT")
ax.set_aspect("equal")
plt.title("Gridded Anomaly Magnetic data from Rio de Janeiro")
plt.show()
