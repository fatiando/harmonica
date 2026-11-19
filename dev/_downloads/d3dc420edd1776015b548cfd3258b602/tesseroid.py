#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boule as bl

ellipsoid = bl.WGS84
mean_radius = ellipsoid.mean_radius


# In[2]:


tesseroid = (-70, -50, -40, -20, mean_radius - 10e3, mean_radius)
density = 2670


# In[3]:


import bordado as bd

coordinates = bd.grid_coordinates(
    region=[-80, -40, -50, -10],
    shape=(80, 80),
    non_dimensional_coords=100e3 + mean_radius,
)


# In[4]:


import harmonica as hm

gravity = hm.tesseroid_gravity(coordinates, tesseroid, density, field="g_z")


# In[5]:


import pygmt

# Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
# the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
# for sphinx-gallery to work means that fig.show won't display anything here
# either.
pygmt.set_display(method="notebook")


# In[6]:


import pygmt
import verde as vd

grid = vd.make_xarray_grid(
   coordinates, gravity, data_names="gravity", extra_coords_names="extra")

fig = pygmt.Figure()
title = "Downward component of gravitational acceleration"
with pygmt.config(FONT_TITLE="12p"):
   fig.grdimage(
      region=[-80, -40, -50, -10],
      projection="M-60/-30/10c",
      grid=grid.gravity,
      frame=["a", f"+t{title}"],
      cmap="viridis",
   )

fig.colorbar(cmap=True, frame=["a200f50", "x+lmGal"])
fig.coast(shorelines="1p,black")

# Plot edges of tesseroid
fig.plot(
   x=[tesseroid[0], tesseroid[1], tesseroid[1], tesseroid[0], tesseroid[0]],
   y=[tesseroid[2], tesseroid[2], tesseroid[3], tesseroid[3], tesseroid[2]],
   pen="1p,red",
   label="Tesseroid boundary",
)
fig.legend()

fig.show()


# In[7]:


tesseroids = [
    [-70, -65, -40, -35, mean_radius - 100e3, mean_radius],
    [-55, -50, -40, -35, mean_radius - 100e3, mean_radius],
    [-70, -65, -25, -20, mean_radius - 100e3, mean_radius],
    [-55, -50, -25, -20, mean_radius - 100e3, mean_radius],
]
densities = [2670 , 2670, 2670, 2670]


# In[8]:


coordinates = bd.grid_coordinates(
    region=[-80, -40, -50, -10],
    shape=(80, 80),
    non_dimensional_coords=100e3 + mean_radius,
)
gravity = hm.tesseroid_gravity(coordinates, tesseroids, densities, field="g_z")


# In[9]:


grid = vd.make_xarray_grid(
   coordinates, gravity, data_names="gravity", extra_coords_names="extra")

fig = pygmt.Figure()
title = "Downward component of gravitational acceleration"
with pygmt.config(FONT_TITLE="12p"):
   fig.grdimage(
      region=[-80, -40, -50, -10],
      projection="M-60/-30/10c",
      grid=grid.gravity,
      frame=["a", f"+t{title}"],
      cmap="viridis",
   )

fig.colorbar(cmap=True, frame=["a1000f500", "x+lmGal"])
fig.coast(shorelines="1p,black")

# Plot edges of tesseroids
for i, tesseroid in enumerate(tesseroids):
   if i == 0:
      label="Tesseroid boundaries"
   else:
      label=None
   fig.plot(
      x=[tesseroid[0], tesseroid[1], tesseroid[1], tesseroid[0], tesseroid[0]],
      y=[tesseroid[2], tesseroid[2], tesseroid[3], tesseroid[3], tesseroid[2]],
      pen="1p,red",
      label=label,
   )
fig.legend()

fig.show()


# In[10]:


tesseroids = (
    [-70, -60, -40, -30, mean_radius - 3e3, mean_radius],
    [-70, -60, -30, -20, mean_radius - 5e3, mean_radius],
    [-60, -50, -40, -30, mean_radius - 7e3, mean_radius],
    [-60, -50, -30, -20, mean_radius - 10e3, mean_radius],
)


# In[11]:


from numba import njit

@njit
def density(radius):
    """Linear density function"""
    top = mean_radius
    bottom = mean_radius - 10e3
    density_top = 2670
    density_bottom = 3000
    slope = (density_top - density_bottom) / (top - bottom)
    return slope * (radius - bottom) + density_bottom


# In[12]:


coordinates = bd.grid_coordinates(
    region=[-80, -40, -50, -10],
    shape=(80, 80),
    non_dimensional_coords=100e3 + ellipsoid.mean_radius,
)


# In[13]:


gravity = hm.tesseroid_gravity(coordinates, tesseroids, density, field="g_z")


# In[14]:


grid = vd.make_xarray_grid(
   coordinates, gravity, data_names="gravity", extra_coords_names="extra")

fig = pygmt.Figure()
title = "Downward component of gravitational acceleration"
with pygmt.config(FONT_TITLE="12p"):
   fig.grdimage(
      region=[-80, -40, -50, -10],
      projection="M-60/-30/10c",
      grid=grid.gravity,
      frame=["a", f"+t{title}"],
      cmap="viridis",
   )
fig.colorbar(cmap=True, frame=["a200f100", "x+lmGal"])
fig.coast(shorelines="1p,black")

# Plot edges of tesseroids
for i, tesseroid in enumerate(tesseroids):
   if i == 0:
      label="Tesseroid boundaries"
   else:
      label=None
   fig.plot(
      x=[tesseroid[0], tesseroid[1], tesseroid[1], tesseroid[0], tesseroid[0]],
      y=[tesseroid[2], tesseroid[2], tesseroid[3], tesseroid[3], tesseroid[2]],
      pen="1p,red",
      label=label,
   )
fig.legend()

fig.show()


# In[15]:


import boule as bl

ellipsoid = bl.WGS84
region = (-80, -40, -50, -10)
spacing = 0.5


# In[16]:


import bordado as bd

longitude, latitude = bd.grid_coordinates(region=region, spacing=spacing)


# In[17]:


reference = ellipsoid.geocentric_radius(latitude)


# In[18]:


import numpy as np

max_height = 3e3
topography = (
    max_height * np.sin(longitude * np.pi / 20) * np.cos(latitude * np.pi / 20)
    + max_height
) / 2
surface = reference + topography


# In[19]:


import verde as vd

surface_grid = vd.make_xarray_grid(
   (longitude, latitude),
   surface,
   data_names="surface",
   dims=("latitude", "longitude"),
)
topography_grid = vd.make_xarray_grid(
   (longitude, latitude),
   topography,
   data_names="topography",
   dims=("latitude", "longitude"),
)

fig = pygmt.Figure()
gmt_projection = "M-60/-30/10c"
title = "Surface boundary of the tesseroid layer"
with pygmt.config(FONT_TITLE="12p"):
   fig.grdimage(
      region=region,
      projection=gmt_projection,
      grid=surface_grid.surface,
      frame=["a", f"+t{title}"],
      cmap="magma",
   )
fig.colorbar(cmap=True, frame=["af", "x+lSurface radius", "y+lmeters"])
fig.coast(shorelines="1p,black")

fig.shift_origin(xshift="w+1.5c")

title = "Topography"
with pygmt.config(FONT_TITLE="12p"):
   fig.grdimage(
      region=region,
      projection=gmt_projection,
      grid=topography_grid.topography,
      frame=["a", f"+t{title}"],
      cmap="magma",
   )
fig.colorbar(cmap=True, frame=["af", "x+lTopography", "y+lmeters"])
fig.coast(shorelines="1p,black")
fig.show()


# In[20]:


density = np.full_like(surface, 2670.0)


# In[21]:


import harmonica as hm

tesseroids = hm.tesseroid_layer(
    coordinates=(longitude, latitude),
    surface=surface,
    reference=reference,
    properties={"density": density},
)
tesseroids


# In[22]:


grid_longitude, grid_latitude = bd.grid_coordinates(region=region, spacing=spacing)
grid_radius = ellipsoid.geocentric_radius(grid_latitude) + 10e3
coordinates = (grid_longitude, grid_latitude, grid_radius)


# In[23]:


gravity = tesseroids.tesseroid_layer.gravity(coordinates, field="g_z")


# In[24]:


grid = vd.make_xarray_grid(
   coordinates,
   gravity,
   data_names="gravity",
   dims=("latitude", "longitude"),
   extra_coords_names="radius",
)

fig = pygmt.Figure()
title = "Gravitational acceleration of a layer of tesseroids"
with pygmt.config(FONT_TITLE="12p"):
   fig.grdimage(
      region=region,
      projection="M-60/-30/10c",
      grid=grid.gravity,
      frame=["a", f"+t{title}"],
      cmap="viridis",
   )
fig.colorbar(cmap=True, frame=["a100f50", "x+lmGal"])
fig.coast(shorelines="1p,black")
fig.show()

