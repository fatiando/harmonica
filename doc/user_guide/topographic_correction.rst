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

The simpler way to apply the topographic correction is through the **Bouguer
correction**. It consists in approximating the topographic masses that lay
underneath each computation point as an infinite slab of constant density.
It has been widely used on ground surveys because it's easy to compute and
because we don't need any other data than the observation height (the height at
which the gravity has been measured).
It's main drawback is it's accuracy: the approximation might be too simple to
accurately reproduce the gravitational effect of the topography present in our
region of interest.

On the other hand, we can compute the topographic correction by **forward
modelling the topographic masses**. To do so we will need a 2D grid of the
topography, a.k.a. a DEM (digital elevation model). This method produces
accurate corrections if the DEM has good resolutions, but its computation is
much more expensive.

In the following sections we will explore how we can apply both methods using
Harmonica.

Lets start by downloading some gravity data over the Bushveld Igneous Complex
in Southern Africa.

.. jupyter-execute::

   import ensaio
   import pandas as pd

   fname = ensaio.fetch_bushveld_gravity(version=1)
   data = pd.read_csv(fname)
   data

And plot it:

.. jupyter-execute::

   import pygmt
   import verde as vd

   maxabs = vd.maxabs(data.gravity_disturbance_mgal)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="polar+h0", series=[-maxabs, maxabs])
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      color=data.gravity_disturbance_mgal,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen'],
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lgravity disturbance", "y+lmGal"])
   fig.show()


Bouguer correction
------------------

We can compute the Bouguer correction through the
:func:`harmonica.bouguer_correction` function.
Because our gravity data has been obtained on the Earth surface, the
``height_geometric_m`` coordinate coincides with the topographic height at each
observation point (referenced above the ellipsoid), so we can pass it as the
``topography`` argument.

.. jupyter-execute::

   import harmonica as hm

   bouguer_correction = hm.bouguer_correction(data.height_geometric_m)

.. hint::

   The :func:`harmonica.bouguer_correction` assigns default values for the
   density of the upper crust and the water.

.. warning::

   In case the observations heights were referenced over the geoid (usually
   marked as above the mean sea level), it's advisable to convert them to
   geometric heights by removing the geoid height.

We can now compute the Bouguer disturbance and plot it:

.. jupyter-execute::

   bouguer_disturbance = data.gravity_disturbance_mgal - bouguer_correction
   bouguer_disturbance

.. jupyter-execute::

   maxabs = vd.maxabs(bouguer_disturbance)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="polar+h0", series=[-maxabs, maxabs])
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      color=bouguer_disturbance,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen'],
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lBouguer disturbance (with simple Bouguer correction)", "y+lmGal"])
   fig.show()



Forward modelling the topography
--------------------------------

In order to forward model the topographic masses, we need to build a 3D model
made out of simpler geometric bodies. In this case, we are going to use
rectangular prisms.
Then we will compute the gravitational effect of every prism on each
computation point.

To do so, we need a regular grid of the topographic heights (or DEM as in
Digital Elevation Model) around the Bushveld Igneous Complex.
We can download a global topography grid:

.. jupyter-execute::

   import xarray as xr

   fname = ensaio.fetch_southern_africa_topography(version=1)
   topography = xr.load_dataarray(fname)
   topography

And then crop it to a slightly larger region than the gravity observations:

.. jupyter-execute::

   region = vd.get_region((data.longitude, data.latitude))
   region_pad = vd.pad_region(region, pad=1)

   topography = topography.sel(
       longitude=slice(region_pad[0], region_pad[1]),
       latitude=slice(region_pad[2], region_pad[3]),
   )
   topography

And project it to plain coordinates using :mod:`pyproj` and :mod:`verde`.
We start by defining a Mercator projection:

.. jupyter-execute::

   import pyproj

   projection = pyproj.Proj(proj="merc", lat_ts=topography.latitude.values.mean())

And project the grid using :func:`verde.project_grid`:

.. jupyter-execute::

   topography_proj = vd.project_grid(topography, projection, method="nearest")
   topography_proj

.. tip::

   Using the ``"nearest"`` method makes the projection process faster than
   using the ``"linear"`` one.

Now we can create a 3D model of the topographic masses using a layer of
rectangular prisms. We can use the :func:`harmonica.prism_layer` function to
build it.
We also need to assign density values to each prism in the layer.
For every prism above the ellipsoid we will set the density of the upper crust
(2670 kg/m\ :sup:`3`), while for each prism below it we will assign the
density contrast equal to the density of the water (1040 kg/m\ :sup:`3`) minus
the density of the upper crust.

.. jupyter-execute::

   import numpy as np

   density = np.where(topography_proj >= 0, 2670, 1040 - 2670)

   prisms = hm.prism_layer(
       (topography_proj.easting, topography_proj.northing),
       surface=topography_proj,
       reference=0,
       properties={"density": density},
   )
   prisms

Now we need to compute the gravitational effect of these prisms on every
observation point. We can do it through the
:meth:`harmonica.DatasetAccessorPrismLayer.gravity` method. But the coordinates
of the observation points must be also projected.

.. jupyter-execute::

   # Project the coordinates of the observation points
   easting, northing = projection(data.longitude.values, data.latitude.values)
   coordinates = (easting, northing, data.height_geometric_m)

   # Compute the terrain effect
   terrain_effect = prisms.prism_layer.gravity(coordinates, field="g_z")

Finally, we can compute the topography-free gravity disturbance:

.. jupyter-execute::

   topo_free_disturbance = data.gravity_disturbance_mgal - terrain_effect

And plot it:

.. jupyter-execute::

   maxabs = vd.maxabs(topo_free_disturbance)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="polar+h0", series=[-maxabs, maxabs])
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      color=topo_free_disturbance,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen'],
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lTopography-free gravity disturbance", "y+lmGal"])
   fig.show()

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <topographic_correction>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <topographic_correction>`
        :text-align: center
