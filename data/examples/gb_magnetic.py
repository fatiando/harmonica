"""
Total Field Magnetic Anomaly from Great Britain
================================================

These data are a complete airborne survey of the entire United Kingdom
conducted between 1955 and 1965. The data are made available by the
British Geological Survey (BGS) through their `geophysical data portal
<https://www.bgs.ac.uk/products/geophysics/aeromagneticRegional.html>`__.
The columns of the data table are longitude, latitude, total-field
magnetic anomaly (nanoTesla), observation height relative to Ordnance Survey
datum (in meters), survey area, and line number and line segment for each data point.
If the file isn't already in your data directory, it will be downloaded
automatically. See the documentation for :func:`harmonica.datasets.fetch_gb_magnetic`
for more information.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd
import harmonica as hm

# Fetch the data in a pandas.DataFrame
data = hm.datasets.fetch_gb_magnetic()
print(data)

# Plot the observations in a Mercator map using Cartopy
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Magnetic data from Great Britain", pad=25)
maxabs = vd.maxabs(data.MAG_IGRF90)
tmp = ax.scatter(
    data.LONGITUDE,
    data.LATITUDE,
    c=data.MAG_IGRF90,
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
ax.set_extent(vd.get_region((data.LONGITUDE, data.LATITUDE)))
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()
