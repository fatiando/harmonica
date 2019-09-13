"""
Land Gravity Data from South Africa
===================================

Land gravity survey performed in January 1986 within the boundaries of the Republic of
South Africa. The data are stored in a :class:`pandas.DataFrame` with columns:
longitude, latitude, elevation (above sea level) and gravity(mGal). See the
documentation for :func:`harmonica.datasets.fetch_south_africa_gravity` for more
information.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd
import harmonica as hm

# Fetch the data in a pandas.DataFrame
data = hm.datasets.fetch_south_africa_gravity()
print(data)

# Plot the observations in a Mercator map using Cartopy
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Observed gravity data from South Africa", pad=25)
tmp = ax.scatter(
    data.longitude,
    data.latitude,
    c=data.gravity,
    s=0.8,
    cmap="viridis",
    transform=ccrs.PlateCarree(),
)
plt.colorbar(
    tmp,
    ax=ax,
    label="observed gravity [mGal]",
    orientation="horizontal",
    aspect=50,
    shrink=0.6,
    pad=0.06,
)
ax.set_extent(vd.get_region((data.longitude, data.latitude)))
ax.gridlines(draw_labels=True)
ax.coastlines()
plt.tight_layout()
plt.show()
