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

projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.values.mean())
easting, northing = projection(data.longitude.values, data.latitude.values)

coordinates = (easting, northing, data.height_geometric_m.values)


# In[3]:


import harmonica as hm

eqs_first_guess = hm.EquivalentSources(depth=1e3, damping=1)
eqs_first_guess.fit(coordinates, data.gravity_disturbance_mgal)


# In[4]:


import numpy as np
import verde as vd

score_first_guess = np.mean(
    vd.cross_val_score(
        eqs_first_guess,
        coordinates,
        data.gravity_disturbance_mgal,
    )
)
score_first_guess


# In[5]:


dampings = [0.01, 0.1, 1, 10,]
depths = [5e3, 10e3, 20e3, 50e3]


# In[6]:


import itertools

parameter_sets = [
    dict(damping=combo[0], depth=combo[1])
    for combo in itertools.product(dampings, depths)
]
print("Number of combinations:", len(parameter_sets))
print("Combinations:", parameter_sets)


# In[7]:


equivalent_sources = hm.EquivalentSources()

scores = []
for params in parameter_sets:
    equivalent_sources.set_params(**params)
    score = np.mean(
        vd.cross_val_score(
            equivalent_sources,
            coordinates,
            data.gravity_disturbance_mgal,
        )
    )
    scores.append(score)
scores


# In[8]:


best = np.argmax(scores)
print("Best score:", scores[best])
print("Score with defaults:", score_first_guess)
print("Best parameters:", parameter_sets[best])


# In[9]:


eqs_best = hm.EquivalentSources(**parameter_sets[best]).fit(
    coordinates, data.gravity_disturbance_mgal
)


# In[10]:


# Define grid coordinates
region = vd.get_region(coordinates)
grid_coords = vd.grid_coordinates(
    region=region,
    spacing=2e3,
    extra_coords=2.5e3,
)

grid_first_guess = eqs_first_guess.grid(grid_coords)
grid = eqs_best.grid(grid_coords)


# In[11]:


import pygmt

# Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
# the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
# for sphinx-gallery to work means that fig.show won't display anything here
# either.
pygmt.set_display(method="notebook")


# In[12]:


import pygmt

# Set figure properties
w, e, s, n = region
fig_height = 10
fig_width = fig_height * (e - w) / (n - s)
fig_ratio = (n - s) / (fig_height / 100)
fig_proj = f"x1:{fig_ratio}"

maxabs = vd.maxabs(grid_first_guess.scalars, grid.scalars)

fig = pygmt.Figure()

# Make colormap of data
pygmt.makecpt(cmap="polar+h0",series=(-maxabs, maxabs,))

title = "Gravity disturbance with first guess"

fig.grdimage(
   projection=fig_proj,
   region=region,
   frame=[f"WSne+t{title}", "xa100000+a15", "ya100000"],
   grid=grid_first_guess.scalars,
   cmap=True,
)
fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

fig.shift_origin(xshift=fig_width + 1)

title = "Gravity disturbance with best params"

fig.grdimage(
   frame=[f"ESnw+t{title}", "xa100000+a15", "ya100000"],
   grid=grid.scalars,
   cmap=True,
)
fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

fig.show()

