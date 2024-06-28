#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ensaio
import pandas as pd

fname = ensaio.fetch_bushveld_gravity(version=1)
data = pd.read_csv(fname)
data


# In[2]:


import pyproj
import verde as vd

projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.values.mean())
easting, northing = projection(data.longitude.values, data.latitude.values)
region = vd.get_region((easting, northing))


# In[3]:


import harmonica as hm

equivalent_sources = hm.EquivalentSources(damping=10)
equivalent_sources


# In[4]:


coordinates = (easting, northing, data.height_geometric_m)
equivalent_sources.fit(coordinates, data.gravity_disturbance_mgal)


# In[5]:


disturbance = equivalent_sources.predict(coordinates)


# In[6]:


import pygmt

# Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
# the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
# for sphinx-gallery to work means that fig.show won't display anything here
# either.
pygmt.set_display(method="notebook")


# In[7]:


import pygmt

# Get max absolute value for the observed gravity disturbance
maxabs = vd.maxabs(data.gravity_disturbance_mgal)

# Set figure properties
w, e, s, n = region
fig_height = 10
fig_width = fig_height * (e - w) / (n - s)
fig_ratio = (n - s) / (fig_height / 100)
fig_proj = f"x1:{fig_ratio}"

fig = pygmt.Figure()
pygmt.makecpt(cmap="polar+h0", series=[-maxabs, maxabs])
title="Predicted gravity disturbance"
with pygmt.config(FONT_TITLE="14p"):
   fig.plot(
      x=easting,
      y=northing,
      fill=disturbance,
      cmap=True,
      style="c3p",
      projection=fig_proj,
      region=region,
      frame=['ag', f"+t{title}"],
   )
fig.colorbar(cmap=True, position="JMR", frame=["a50f25", "y+lmGal"])

fig.shift_origin(yshift=fig_height + 2)

title="Observed gravity disturbance"
with pygmt.config(FONT_TITLE="14p"):
   fig.plot(
      x=easting,
      y=northing,
      fill=data.gravity_disturbance_mgal,
      cmap=True,
      style="c3p",
      frame=['ag', f"+t{title}"],
   )
fig.colorbar(cmap=True, position="JMR", frame=["a50f25", "y+lmGal"])

fig.show()


# In[8]:


data.height_geometric_m.max()


# In[9]:


# Build the grid coordinates
grid_coords = vd.grid_coordinates(region=region, spacing=2e3, extra_coords=2.2e3)

# Grid the gravity disturbances
grid = equivalent_sources.grid(grid_coords, data_names=["gravity_disturbance"])
grid


# In[10]:


maxabs = vd.maxabs(grid.gravity_disturbance)

fig = pygmt.Figure()
pygmt.makecpt(cmap="polar+h0", series=[-maxabs, maxabs])
fig.grdimage(
   frame=['af', 'WSen'],
   grid=grid.gravity_disturbance,
   region=region,
   projection=fig_proj,
   cmap=True,
)
fig.colorbar(cmap=True, frame=["a50f25", "x+lgravity disturbance", "y+lmGal"])

fig.show()

