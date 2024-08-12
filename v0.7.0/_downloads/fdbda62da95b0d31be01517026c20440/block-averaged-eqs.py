#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ensaio
import pandas as pd

fname = ensaio.fetch_britain_magnetic(version=1)
data = pd.read_csv(fname)
data


# In[2]:


import verde as vd

region = (-5.5, -4.7, 57.8, 58.5)
inside = vd.inside((data.longitude, data.latitude), region)
data = data[inside]
data


# In[3]:


import pyproj

projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
easting, northing = projection(data.longitude.values, data.latitude.values)
coordinates = (easting, northing, data.height_m)
xy_region=vd.get_region(coordinates)


# In[4]:


import pygmt

# Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
# the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
# for sphinx-gallery to work means that fig.show won't display anything here
# either.
pygmt.set_display(method="notebook")


# In[5]:


import pygmt

maxabs = vd.maxabs(data.total_field_anomaly_nt)*.8

# Set figure properties
w, e, s, n = xy_region
fig_height = 15
fig_width = fig_height * (e - w) / (n - s)
fig_ratio = (n - s) / (fig_height / 100)
fig_proj = f"x1:{fig_ratio}"

# Plot original magnetic anomaly and the gridded and upward-continued version
fig = pygmt.Figure()

title = "Observed total-field magnetic anomaly"

pygmt.makecpt(
    cmap="polar+h0",
    series=(-maxabs, maxabs),
    background=True,
)

with pygmt.config(FONT_TITLE="12p"):
    fig.plot(
        projection=fig_proj,
        region=xy_region,
        frame=[f"WSne+t{title}", "xa10000", "ya10000"],
        x=easting,
        y=northing,
        fill=data.total_field_anomaly_nt,
        style="c0.1c",
        cmap=True,
    )
fig.colorbar(cmap=True, position="JMR", frame=["a200f100", "x+lnT"])
fig.show()


# In[6]:


import harmonica as hm

eqs = hm.EquivalentSources(depth=1000, damping=1, block_size=500)


# In[7]:


eqs.fit(coordinates, data.total_field_anomaly_nt)


# In[8]:


eqs.points_[0].size


# In[9]:


grid_coords = vd.grid_coordinates(
    region=vd.get_region(coordinates),
    spacing=500,
    extra_coords=1500,
)
grid = eqs.grid(grid_coords, data_names=["magnetic_anomaly"])
grid


# In[10]:


fig = pygmt.Figure()

title = "Observed magnetic anomaly data"
pygmt.makecpt(
    cmap="polar+h0",
    series=(-maxabs, maxabs),
    background=True)

with pygmt.config(FONT_TITLE="14p"):
    fig.plot(
        projection=fig_proj,
        region=xy_region,
        frame=[f"WSne+t{title}", "xa10000", "ya10000"],
        x=easting,
        y=northing,
        fill=data.total_field_anomaly_nt,
        style="c0.1c",
        cmap=True,
    )
fig.colorbar(cmap=True, frame=["a200f100", "x+lnT"])

fig.shift_origin(xshift=fig_width + 1)

title = "Gridded and upward-continued"

with pygmt.config(FONT_TITLE="14p"):
    fig.grdimage(
        frame=[f"ESnw+t{title}", "xa10000", "ya10000"],
        grid=grid.magnetic_anomaly,
        cmap=True,
    )
fig.colorbar(cmap=True, frame=["a200f100", "x+lnT"])

fig.show()

