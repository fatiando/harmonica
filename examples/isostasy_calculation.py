"""
Airy Isostasy
=============

Calculation of the compensated root/antiroot of the topographic structure assuming the
Airy hypothesis.
If you want to obtain the isostatic Moho, you need to assume a normal crust value.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the elevation model
data = hm.datasets.fetch_topography_earth()
print(data)
data_africa = data.sel(latitude=slice(-40, 30), longitude=slice(-20, 60))
print(data_africa)
# Root calculation considering the ocean
root = hm.isostasy_airy(
    data_africa.topography.values,
    density_upper_crust=2670,
    density_lower_crust=2800,
    density_mantle=3300,
    density_oceanic_crust=2900,
    density_water=1000,
)
data_africa["root"] = (data_africa.dims, root)

# Root calculation without considering the ocean
root_without_ocean = hm.isostasy_airy(
    data_africa.topography.values,
    density_upper_crust=2670,
    density_lower_crust=2800,
    density_mantle=3300,
)
data_africa["root_without_ocean"] = (data_africa.dims, root_without_ocean)

# To obtain the depth of the Moho is necessary to assume a normal crust value
T = 35000
data_africa["moho"] = -(T + data_africa.root)
data_africa["moho_without_ocean"] = -(T + data_africa.root_without_ocean)

# Make maps of both versions using an Albers Equal Area projection
proj = ccrs.AlbersEqualArea(central_longitude=20, central_latitude=0)
trans = ccrs.PlateCarree()

# Setup some common arguments for the colorbar and pseudo-color plot
cbar_kwargs = dict(pad=0, orientation="horizontal")
pcolor_args = dict(cmap="viridis", add_colorbar=False, transform=ccrs.PlateCarree())

# Draw the maps
fig, axes = plt.subplots(1, 2, figsize=(13, 9), subplot_kw=dict(projection=proj))
fig.suptitle("Isostatic Moho using ETOPO1 and Airy hypothesis")
ax = axes[0]
tmp = data_africa.moho.plot.pcolormesh(ax=ax, **pcolor_args)
plt.colorbar(tmp, ax=ax, **cbar_kwargs).set_label("[meters]")
ax.gridlines()
ax.set_title("Moho depth")
ax = axes[1]
tmp = data_africa.moho_without_ocean.plot.pcolormesh(ax=ax, **pcolor_args)
plt.colorbar(tmp, ax=ax, **cbar_kwargs).set_label("[meters]")
ax.gridlines()
ax.set_title("Moho depth without considering the ocean")
plt.tight_layout()
plt.show()
