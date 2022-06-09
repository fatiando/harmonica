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
import pygmt
import verde as vd
import harmonica as hm

# Fetch the data in a pandas.DataFrame
data = hm.datasets.fetch_south_africa_gravity()
print(data)

# Get the region of the grid
region = vd.get_region((data.longitude.values, data.latitude.values))

# Make a plot of data using PyGMT
fig = pygmt.Figure()

title = "Observed gravity data from South Africa"   

pygmt.makecpt(cmap='viridis', series=(data.gravity.min(),data.gravity.max()))

fig.plot(
    region=region,
    projection='M15c',
    frame=['ag', f'+t{title}'],
    x=data.longitude, 
    y=data.latitude, 
    color=data.gravity, 
    style="c0.1c",
    cmap=True,
    )

fig.coast(shorelines='1p,black')

fig.colorbar(cmap=True, frame=['a200f50', 'x+lobserved gravity [mGal]'], position="JMR")

fig.show()