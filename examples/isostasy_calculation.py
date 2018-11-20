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

# Root calculation considering the ocean
root = hm.isostasy_airy(data.topography.values,
                        density_upper_crust=2670,
                        density_lower_crust=2800,
                        density_mantle=3300,
                        density_oceanic_crust=2900,
                        density_water=1000)
data["root"] = (data.dims, root)

# To obtain the depth of the Moho is necessary to assume a normal crust value
T = 35000
data["moho"] = - (T + data.root)


# Make maps of both versions using Cartopy
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=67.5)
trans = ccrs.PlateCarree()

# Setup some common arguments for the colorbar and pseudo-color plot
cbar_kwargs = dict(pad=0, orientation="horizontal")
pcolor_args = dict(cmap="viridis", add_colorbar=False, transform=ccrs.PlateCarree())

# Draw the maps
fig, axes = plt.subplots(1, 2, figsize=(9, 5), subplot_kw=dict(projection=proj))
ax = axes[0]
tmp = data.root.plot.pcolormesh(ax=ax, **pcolor_args)
plt.colorbar(tmp, ax=ax, **cbar_kwargs).set_label("[meters]")
ax.gridlines()
ax.set_title("Isostatic root")
ax = axes[1]
tmp = data.moho.plot.pcolormesh(ax=ax, **pcolor_args)
plt.colorbar(tmp, ax=ax, **cbar_kwargs).set_label("[meters]")
ax.gridlines()
ax.set_title("Isostatic depth Moho")
plt.tight_layout()
plt.show()
