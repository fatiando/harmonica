.. _topographic_correction:

Topographic Correction
======================

Computing the :ref:`gravity disturbance <gravity_disturbance>` is usually the
first step towards generating a dataset that could provide insight of the
structures and bodies that lie beneath Earth surface.

One of the strongest signals present in the gravity disturbances are the
gravitational effect of the topography, i.e. every body located above the
surface of the reference ellipsoid.
Mainly because their proximity to the observation points but also because their
density contrast could be considered the same as their own absolute density.

For this reason, geophysicists usually remove the gravitational effect of the
topography from the gravity disturbance in a processes called **topographic
correction**.
The resulting field is often called *Bouguer gravity disturbance* or
*topography-free gravity disturbance*.

The simpler way to apply the topographic correction is through the *Bouguer
correction*. It consists in approximating the topographic masses that lay
underneath each computation point as an infinite slab of constant density.
It has been widely used on ground surveys because it's easy to compute and
because we don't need any other data than the observation height (the height at
which the gravity has been measured).
It's main drawback is it's accuracy: the approximation might be too simple to
accurately reproduce the gravitational effect of the topography present in our
region of interest.

On the other hand, we can compute the topographic correction by forward
modelling the topographic masses. To do so we will need a 2D grid of the
topography, a.k.a. a DEM (digital elevation model). This method produces
accurate corrections if the DEM has good resolutions, but its computation is
much more expensive.

In the following sections we will explore how we can apply both methods using
Harmonica.
Lets start by computing the gravity disturbance from a ground gravity survey
over Southern Africa. We can start by downloading the gravity dataset:

.. jupyter-execute::

   import ensaio
   import pandas as pd

   fname = ensaio.fetch_southern_africa_gravity(version=1)
   data = pd.read_csv(fname)

Compute the normal gravity through :mod:`boule`:

.. jupyter-execute::

   import boule as bl

   ellipsoid = bl.WGS84
   normal_gravity = ellipsoid.normal_gravity(data.latitude, data.height_sea_level_m)

And calculate the gravity disturbance:

.. jupyter-execute::

   gravity_disturbance = data.gravity_mgal - normal_gravity
   gravity_disturbance

.. jupyter-execute::

   import cartopy.crs as ccrs
   import matplotlib.pyplot as plt
   import verde as vd

   maxabs = vd.maxabs(gravity_disturbance)

   fig = plt.figure(figsize=(10, 8), dpi=300)
   ax = plt.axes(projection=ccrs.Mercator())
   ax.set_title("Gravity disturbance", pad=25)
   tmp = ax.scatter(
       data.longitude,
       data.latitude,
       c=gravity_disturbance,
       s=1.5,
       vmin=-maxabs,
       vmax=maxabs,
       cmap="seismic",
       transform=ccrs.PlateCarree(),
   )
   plt.colorbar(
       tmp, ax=ax, label="mGal", aspect=50, pad=0.1, shrink=0.92
   )
   ax.set_extent(vd.get_region((data.longitude, data.latitude)))
   ax.gridlines(draw_labels=True)
   ax.coastlines()
   plt.show()


Bouguer correction
------------------

We can compute the Bouguer correction through the
:func:`harmonica.bouguer_correction` function.
Because our gravity data has been obtained on the Earth surface, the
``height_sea_level_m`` coordinate coincides with the topographic height at each
observation point, so we can pass it as the ``topography`` argument.

.. jupyter-execute::

   import harmonica as hm

   bouguer_correction = hm.bouguer_correction(data.height_sea_level_m)

.. hint::

   The :func:`harmonica.bouguer_correction` assigns default values for the
   density of the upper crust and the water.

.. warning::

   The ``height_sea_level_m`` array in this particular dataset is referenced to
   the *mean sea-level*, which means that their values are not geodetic
   heights, but above the geoid. In this example we are going to ignore the
   differences generated by the geoid undulation for simplicity, but we
   recommend adjusting the heights in a real world problem.

We can now compute the Bouguer disturbance and plot it:

.. jupyter-execute::

   bouguer_disturbance = gravity_disturbance - bouguer_correction
   bouguer_disturbance

.. jupyter-execute::

   maxabs = vd.maxabs(bouguer_disturbance)

   fig = plt.figure(figsize=(10, 8), dpi=300)
   ax = plt.axes(projection=ccrs.Mercator())
   ax.set_title("Bouguer disturbance (with simple Bouguer correction)", pad=25)
   tmp = ax.scatter(
       data.longitude,
       data.latitude,
       c=bouguer_disturbance,
       s=1.5,
       vmin=-maxabs,
       vmax=maxabs,
       cmap="seismic",
       transform=ccrs.PlateCarree(),
   )
   plt.colorbar(
       tmp, ax=ax, label="mGal", aspect=50, pad=0.1, shrink=0.92
   )
   ax.set_extent(vd.get_region((data.longitude, data.latitude)))
   ax.gridlines(draw_labels=True)
   ax.coastlines()
   plt.show()



Forward modelling the topography
--------------------------------

We will create a model of a topography grid for out region of interest made out
of rectangular prisms, and then we will use it to compute its gravitational
effect on each observation point.

Lets start by fetching a topography grid:

.. jupyter-execute::

    topography = hm.datasets.fetch_south_africa_topography().topography
    topography

We will build the prism model using the :ref:`prism layer <prism_layer>`.
Because prisms need to be defined in Cartesian coordinates, we need to project
the topography grid and also our data points.
Lets define a Mercator projection using :mod:`pyproj`:

.. jupyter-execute::

   import pyproj

   projection = pyproj.Proj(proj="merc", lat_ts=topography.latitude.values.mean())

Then, project the data points:

.. jupyter-execute::

   easting, northing = projection(data.longitude.values, data.latitude.values)
   data = data.assign(easting=easting, northing=northing)
   data

And use :func:`verde.project_grid` to project the topography grid we
downloaded:

.. jupyter-execute::

   topography_proj = vd.project_grid(topography, projection)
   topography_proj

Once our topography grid is defined in Cartesian coordinates, we can build
a prism layer through :func:`harmonica.prism_layer`.
The ``surface`` of the layer will be equal to the topography grid, while the
``reference`` will be set to the zeroth height. The density of the prisms will
be assigned according to the elevation of the topography points:

- for points in the continent (positive height), we will assign a density of
  2670 kg per cubic meter.
- for points in the ocean (negative height obtained through bathymetry), we
  will assign a density contrast of (1000 - 2900) kg per cubic meter (density
  of the water minus the density of the upper crust).


.. jupyter-execute::

   density = topography_proj.copy()
   density.values[:] = 2670
   density = density.where(topography_proj >= 0, 1000 - 2900)

   prisms = hm.prism_layer(
       (topography_proj.easting, topography_proj.northing),
       surface=topography_proj,
       reference=0,
       properties={"density": density},
   )
   prisms

Now we can compute the gravitational effect of these prisms on the observation
points:

.. jupyter-execute::

   topographic_correction = prisms.prism_layer.gravity(
       (data.easting, data.northing, data.height_sea_level_m), field="g_z"
   )

.. jupyter-execute::

   topography_free_disturbance = gravity_disturbance - topographic_correction
   topography_free_disturbance

.. jupyter-execute::

   maxabs = vd.maxabs(topography_free_disturbance)

   fig = plt.figure(figsize=(10, 8), dpi=300)
   ax = plt.axes(projection=ccrs.Mercator())
   ax.set_title("Bouguer disturbance (through modelling topography)", pad=25)
   tmp = ax.scatter(
       data.longitude,
       data.latitude,
       c=topography_free_disturbance,
       s=1.5,
       vmin=-maxabs,
       vmax=maxabs,
       cmap="seismic",
       transform=ccrs.PlateCarree(),
   )
   plt.colorbar(
       tmp, ax=ax, label="mGal", aspect=50, pad=0.1, shrink=0.92
   )
   ax.set_extent(vd.get_region((data.longitude, data.latitude)))
   ax.gridlines(draw_labels=True)
   ax.coastlines()
   plt.show()