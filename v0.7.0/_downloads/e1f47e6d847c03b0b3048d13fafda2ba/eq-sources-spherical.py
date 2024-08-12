#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ensaio
import pandas as pd

fname = ensaio.fetch_southern_africa_gravity(version=1)
data = pd.read_csv(fname)
data


# In[2]:


import numpy as np
import verde as vd

blocked_mean = vd.BlockReduce(np.mean, spacing=6 / 60, drop_coords=False)
(longitude, latitude, height), gravity_data = blocked_mean.filter(
    (data.longitude, data.latitude, data.height_sea_level_m),
    data.gravity_mgal,
)

data = pd.DataFrame(
    {
        "longitude": longitude,
        "latitude": latitude,
        "height_sea_level_m": height,
        "gravity_mgal": gravity_data,
    }
)
data


# In[3]:


import boule as bl

ellipsoid = bl.WGS84
normal_gravity = ellipsoid.normal_gravity(data.latitude, data.height_sea_level_m)
gravity_disturbance = data.gravity_mgal - normal_gravity


# In[4]:


import harmonica as hm

eqs = hm.EquivalentSourcesSph(damping=1e-3, relative_depth=10000)


# In[5]:


coordinates = ellipsoid.geodetic_to_spherical(
    data.longitude, data.latitude, data.height_sea_level_m
)


# In[6]:


eqs.fit(coordinates, gravity_disturbance)


# In[7]:


# Get the bounding region of the data in geodetic coordinates
region = vd.get_region((data.longitude, data.latitude))

# Get the maximum height of the data coordinates
max_height = data.height_sea_level_m.max()

# Define a regular grid of points in geodetic coordinates
grid_coords = vd.grid_coordinates(
    region=region, spacing=6 / 60, extra_coords=max_height
)


# In[8]:


grid_coords_sph = ellipsoid.geodetic_to_spherical(*grid_coords)


# In[9]:


gridded_disturbance = eqs.predict(grid_coords_sph)


# In[10]:


grid = vd.make_xarray_grid(
    grid_coords,
    gridded_disturbance,
    data_names=["gravity_disturbance"],
    extra_coords_names="upward",
)
grid


# In[11]:


grid = vd.distance_mask(
    data_coordinates=(data.longitude, data.latitude), maxdist=0.5, grid=grid
)


# In[12]:


import pygmt

# Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
# the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
# for sphinx-gallery to work means that fig.show won't display anything here
# either.
pygmt.set_display(method="notebook")


# In[13]:


import pygmt

maxabs = vd.maxabs(gravity_disturbance, grid.gravity_disturbance.values)

fig = pygmt.Figure()

# Make colormap of data
pygmt.makecpt(cmap="polar+h0",series=(-maxabs, maxabs,))

title = "Block-median reduced gravity disturbance"
fig.plot(
    projection="M100/15c",
    region=region,
    frame=[f"WSne+t{title}", "xa5", "ya4"],
    x=longitude,
    y=latitude,
    fill=gravity_disturbance,
    style="c0.1c",
    cmap=True,
)
fig.coast(shorelines="0.5p,black", area_thresh=1e4)
fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

fig.shift_origin(xshift='w+3c')

title = "Gridded gravity disturbance"
fig.grdimage(
    grid=grid.gravity_disturbance,
    cmap=True,
    frame=[f"ESnw+t{title}","xa5", "ya4"],
    nan_transparent=True,
)
fig.coast(shorelines="0.5p,black", area_thresh=1e4)
fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

fig.show()

