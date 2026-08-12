#!/usr/bin/env python
# coding: utf-8

# In[1]:


import harmonica as hm


# In[2]:


prism = (-100, 100, -100, 100, -1000, 500)
density = 3000


# In[3]:


coordinates = (0, 0, 1000)


# In[4]:


potential = hm.prism_gravity(coordinates, prism, density, field="potential")
print(potential, "J/kg")


# In[5]:


import bordado as bd

region = (-10e3, 10e3, -10e3, 10e3)
shape = (51, 51)
height = 10
coordinates = bd.grid_coordinates(
    region, shape=shape, non_dimensional_coords=height
)


# In[6]:


prism = [-2e3, 2e3, -2e3, 2e3, -1.6e3, -900]
density = 3300


# In[7]:


fields = (
   "potential",
   "g_e", "g_n", "g_z",
   "g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"
)

results = {}
for field in fields:
   results[field] = hm.prism_gravity(coordinates, prism, density, field=field)


# In[8]:


import verde as vd

grid = vd.make_xarray_grid(
   coordinates,
   tuple(results.values()),
   data_names=results.keys(),
   extra_coords_names="extra",
)
grid


# In[9]:


import matplotlib.pyplot as plt

grid.potential.plot(cbar_kwargs={"label":"J/kg"})
plt.gca().set_aspect("equal")
plt.show()


# In[10]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))

fields = ["g_e", "g_n", "g_z"]
maxabs = vd.maxabs(*[grid[f] for f in fields], percentile=99)
for field, ax in zip(fields, axes):
   tmp = grid[field].plot(
      ax=ax,
      add_colorbar=False,
      vmin=-maxabs,
      vmax=maxabs,
      cmap='RdBu_r',
   )
   ax.set_aspect("equal")
   ax.set_title(field)
fig.colorbar(tmp, ax=axes.ravel().tolist(), orientation="horizontal", label="mGal", fraction=.03, pad=0.08)
plt.show()


# In[11]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))

fields = ["g_ee", "g_nn", "g_zz"]
maxabs = vd.maxabs(*[grid[f] for f in fields], percentile=99)
for field, ax in zip(fields, axes):
   tmp = grid[field].plot(
      ax=ax,
      add_colorbar=False,
      vmin=-maxabs,
      vmax=maxabs,
      cmap='RdBu_r',
   )
   ax.set_aspect("equal")
   ax.set_title(field)
fig.colorbar(tmp, ax=axes.ravel().tolist(), orientation="horizontal", label="Eotvos", fraction=.03, pad=0.08)
plt.show()


# In[12]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))

fields = ["g_en", "g_ez", "g_nz"]
maxabs = vd.maxabs(*[grid[f] for f in fields], percentile=99)
for field, ax in zip(fields, axes):
   tmp = grid[field].plot(
      ax=ax,
      add_colorbar=False,
      vmin=-maxabs,
      vmax=maxabs,
      cmap='RdBu_r',
   )
   ax.set_aspect("equal")
   ax.set_title(field)
fig.colorbar(tmp, ax=axes.ravel().tolist(), orientation="horizontal", label="Eotvos", fraction=.03, pad=0.08)
plt.show()


# In[13]:


prisms = [
    [2e3, 3e3, 2e3, 3e3, -10e3, -1e3],
    [3e3, 4e3, 7e3, 8e3, -9e3, -1e3],
    [7e3, 8e3, 1e3, 2e3, -7e3, -1e3],
    [8e3, 9e3, 6e3, 7e3, -8e3, -1e3],
]
densities = [2670, 3300, 2900, 2980]


# In[14]:


import bordado as bd

coordinates = bd.grid_coordinates(
    region=(0, 10e3, 0, 10e3), shape=(40, 40), non_dimensional_coords=0
)


# In[15]:


g_z = hm.prism_gravity(coordinates, prisms, densities, field="g_z")


# In[16]:


grid = vd.make_xarray_grid(
   coordinates, g_z, data_names="g_z", extra_coords_names="extra")
grid.g_z.plot(cbar_kwargs={"label":"mGal"})


# In[17]:


import numpy as np

prisms = [
    [-5e3, -3e3, -5e3, -2e3, -10e3, -1e3],
    [3e3, 4e3, 4e3, 5e3, -9e3, -1e3],
]

magnetization_easting = np.array([0.5, -0.4])
magnetization_northing = np.array([0.5, 0.3])
magnetization_upward = np.array([-0.78989, 0.2])
magnetization = (
   magnetization_easting, magnetization_northing, magnetization_upward
)


# In[18]:


region = (-10e3, 10e3, -10e3, 10e3)
shape = (51, 51)
height = 10
coordinates = bd.grid_coordinates(
    region, shape=shape, non_dimensional_coords=height
)


# In[19]:


b_e, b_n, b_u = hm.prism_magnetic(coordinates, prisms, magnetization, field="b")


# In[20]:


grid = vd.make_xarray_grid(
   coordinates,
   data=(b_e, b_n, b_u),
   data_names=["b_e", "b_n", "b_u"],
   extra_coords_names="extra"
)


# In[21]:


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
   grid[field].plot.contour(
      ax=ax,
      colors="k",
      linewidths=0.5,
   )
   ax.set_aspect("equal")
   ax.set_title(field)
fig.colorbar(tmp, ax=axes.ravel().tolist(), orientation="horizontal", label="nT", fraction=.03, pad=0.08)
plt.show()


# In[22]:


b_u = hm.prism_magnetic(
   coordinates, prisms, magnetization, field="b_u"
)


# In[23]:


maxabs = vd.maxabs(b_u)

tmp = plt.pcolormesh(
   coordinates[0], coordinates[1], b_u, vmin=-maxabs, vmax=maxabs, cmap="RdBu_r"
)
plt.contour(coordinates[0], coordinates[1], b_u, colors="k", linewidths=0.5)
plt.title("Bu")
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="nT", pad=0.03, shrink=0.8)
plt.show()


# In[24]:


region = (0, 100e3, -40e3, 40e3)
spacing = 2000


# In[25]:


easting, northing = bd.grid_coordinates(region=region, spacing=spacing)


# In[26]:


wavelength = 24 * spacing
surface = np.abs(np.sin(easting * 2 * np.pi / wavelength))


# In[27]:


density = np.full_like(surface, 2700)


# In[28]:


prisms = hm.prism_layer(
    coordinates=(easting, northing),
    surface=surface,
    reference=0,
    properties={"density": density},
)


# In[29]:


region_pad = bd.pad_region(region, 10e3)
coordinates = bd.grid_coordinates(
    region_pad, spacing=spacing, non_dimensional_coords=1e3
)


# In[30]:


gravity = prisms.prism_layer.gravity(coordinates, field="g_z")


# In[31]:


grid = vd.make_xarray_grid(
   coordinates, gravity, data_names="gravity", extra_coords_names="extra")

grid.gravity.plot(cbar_kwargs={"label":"mGal"})

