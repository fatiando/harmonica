#!/usr/bin/env python
# coding: utf-8

# In[1]:


import harmonica as hm


# In[2]:


dipole = (20, 40, -50)
magnetic_moment = (100, 100, 100)


# In[3]:


coordinates = (20, 40, 10)


# In[4]:


b_e, b_n, b_u = hm.dipole_magnetic(coordinates, dipole, magnetic_moment, field="b")
print(b_e, b_n, b_u)


# In[5]:


b_u = hm.dipole_magnetic(
   coordinates, dipole, magnetic_moment, field="b_u"
)
print(b_u)


# In[6]:


import bordado as bd

region = (-100, 100, -100, 100)
spacing = 1
height = 0
coordinates = bd.grid_coordinates(
   region=region, spacing=spacing, non_dimensional_coords=height
)


# In[7]:


import numpy as np

easting = [25, 35, -30, -50]
northing = [3, -38, 22, -30]
upward = [-200, -100, -300, -150]
dipoles = (easting, northing, upward)

mag_e = [1e3, 2e3, 500, 2e3]
mag_n = [1e3, 2e3, 500, 2e3]
mag_u = [1e3, 2e3, 500, 2e3]
magnetic_moments = (mag_e, mag_n, mag_u)


# In[8]:


b_e, b_n, b_u = hm.dipole_magnetic(coordinates, dipoles, magnetic_moments, field="b")


# In[9]:


import verde as vd

grid = vd.make_xarray_grid(
   coordinates,
   data=(b_e, b_n, b_u),
   data_names=["b_e", "b_n", "b_u"],
   extra_coords_names="extra"
)


# In[10]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))

maxabs = vd.maxabs(*[grid[v] for v in grid.variables], percentile=99)
for field, ax in zip(grid.variables, axes):
   tmp = grid[field].plot(
      ax=ax,
      add_colorbar=False,
      vmin=-maxabs,
      vmax=maxabs,
      cmap='RdBu_r',
   )
   ax.set_aspect("equal")
   ax.set_title(field)
fig.colorbar(tmp, ax=axes.ravel().tolist(), orientation="horizontal", label="nT", fraction=.03, pad=0.08)
plt.show()

