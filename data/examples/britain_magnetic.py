# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Total Field Magnetic Anomaly from Great Britain
================================================

These data are a complete airborne survey of the entire Great Britain
conducted between 1955 and 1965. The data are made available by the
British Geological Survey (BGS) through their `geophysical data portal
<https://www.bgs.ac.uk/products/geophysics/aeromagneticRegional.html>`__.

License: `Open Government License
<http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>`__

The columns of the data table are longitude, latitude, total-field
magnetic anomaly (nanoTesla), observation height relative to the WGS84 datum
(in meters), survey area, and line number and line segment for each data point.

Latitude, longitude, and elevation data converted from original OSGB36
(epsg:27700) coordinate system to WGS84 (epsg:4326) using to_crs function in
GeoPandas.

See the original data for more processing information.

If the file isn't already in your data directory, it will be downloaded
automatically.
"""
import pygmt
import verde as vd
import numpy as np

import harmonica as hm

# Fetch the data in a pandas.DataFrame
data = hm.datasets.fetch_britain_magnetic()
print(data)

# Get the region of the data
region = vd.get_region((data.longitude, data.latitude))

# Plot the observations in a Mercator map using PyGMT
fig = pygmt.Figure()

title = "Magnetic data from Great Britain"   

# Make colormap of data
maxabs = np.percentile(data.total_field_anomaly_nt, 99)
pygmt.makecpt(cmap='polar', series=(-maxabs, maxabs))

fig.plot(
    region=region,
    projection='M10c',
    frame=['ag', f'+t{title}'],
    x=data.longitude, 
    y=data.latitude, 
    color=data.total_field_anomaly_nt, 
    style="c0.2c",
    cmap=True)

fig.colorbar(
    cmap=True, 
    position="JMR",
    frame=['a200f50', 'x+ltotal field magnetic anomaly [nT]'])

fig.coast(shorelines='1p,black')

fig.show()