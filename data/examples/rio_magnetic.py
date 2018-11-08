"""
Total Field Magnetic Anomaly from Rio de Janeiro
================================================

A subsection from an airborne survey of the state of Rio de Janeiro, Brazil, conducted
in 1978. The data are stored in a :class:`pandas.DataFrame` with columns: longitude,
latitude, total field anomaly (nanoTesla), observation height above the WGS84 ellipsoid
(meters), type of flight line (LINE or TIE), and flight line number. See the
documentation for :func:`harmonica.datasets.fetch_rio_magnetic` for more information.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd
import harmonica as hm

# Fetch the data in a pandas.DataFrame
data = hm.datasets.fetch_rio_magnetic()
print(data)

# Plot the observations in a Mercator map using Cartopy
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Magnetic data from Rio de Janeiro", pad=25)
maxabs = vd.maxabs(data.total_field_anomaly_nt)
tmp = ax.scatter(
    data.longitude,
    data.latitude,
    c=data.total_field_anomaly_nt,
    s=0.8,
    cmap="seismic",
    vmin=-maxabs,
    vmax=maxabs,
    transform=ccrs.PlateCarree(),
)
plt.colorbar(
    tmp,
    ax=ax,
    label="total field magnetic anomaly [nT]",
    orientation="horizontal",
    aspect=50,
    shrink=0.7,
    pad=0.06,
)
ax.set_extent(vd.get_region((data.longitude, data.latitude)))
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()
