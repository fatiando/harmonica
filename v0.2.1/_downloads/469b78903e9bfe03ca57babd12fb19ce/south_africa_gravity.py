# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Land Gravity Data from South Africa
===================================

Land gravity survey performed in January 1986 within the boundaries of the
Republic of South Africa. The data was made available by the `National Centers
for Environmental Information (NCEI) <https://www.ngdc.noaa.gov/>`__ (formerly
NGDC) and are in the `public domain
<https://www.ngdc.noaa.gov/ngdcinfo/privacy.html#copyright-notice>`__. The
entire dataset is stored in a :class:`pandas.DataFrame` with columns:
longitude, latitude, elevation (above sea level) and gravity(mGal). See the
documentation for :func:`harmonica.datasets.fetch_south_africa_gravity` for
more information.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd
import harmonica as hm

# Fetch the data in a pandas.DataFrame
data = hm.datasets.fetch_south_africa_gravity()
print(data)

# Plot the observations in a Mercator map using Cartopy
fig = plt.figure(figsize=(6.5, 5))
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
    tmp, ax=ax, label="observed gravity [mGal]", aspect=50, pad=0.1, shrink=0.92
)
ax.set_extent(vd.get_region((data.longitude, data.latitude)))
ax.gridlines(draw_labels=True)
ax.coastlines()
plt.show()
