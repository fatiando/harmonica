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
   :hide-code:

    import pygmt

    # Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
    # the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
    # for sphinx-gallery to work means that fig.show won't display anything here
    # either.
    pygmt.set_display(method="notebook")


.. jupyter-execute::

   import pygmt
   import verde as vd

   maxabs = vd.maxabs(data.gravity_disturbance_mgal)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs])
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      fill=data.gravity_disturbance_mgal,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen+ggray'],
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lGravity disturbance", "y+lmGal"])
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

   cpt_lims = vd.minmax(bouguer_disturbance)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="viridis", series=cpt_lims)
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      fill=bouguer_disturbance,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen+ggray'],
   )
   fig.colorbar(
      cmap=True,
      frame=["a50f25", "x+lBouguer disturbance (with simple Bouguer correction)", "y+lmGal"],
   )
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

   import bordado as bd

   region = bd.get_region((data.longitude, data.latitude))
   region_pad = bd.pad_region(region, pad=1)

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

   cpt_lims = vd.minmax(topo_free_disturbance)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="viridis", series=cpt_lims)
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      fill=topo_free_disturbance,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen+ggray'],
   )
   fig.colorbar(cmap=True, frame=["a50f25", "x+lTopography-free gravity disturbance", "y+lmGal"])
   fig.show()

Compare the Bouguer and Topography-free disturbances
----------------------------------------------------

Now that we have computed the Bouguer and Topography-free disturbances, we
can plot the difference to get a sense of how much the methods differ.
From the below plot, we can see large differences (up to 10 mGal) arise from the
different methods. Generally, these large differences are in regions of rugged
topography, while the regions of flat topography have smaller  differences (<2 mGal).
Additionally, almost all of the differences are positive, meaning the Bouguer
disturbance is generally smaller than the Topography-free disturbance.

This is because the flat-slab assumption always overestimates the correction. It
doesn't account for valleys below or terrain above the observation points, both
of which decrease the observed gravity. Not accounting for these results in too
large of a Bouguer correction, and therefore too small of a Bouguer disturbance.
This highlights the benefit of the forward modelling
topography instead of the Bouguer correction, especially for regions of rugged
topography.

.. jupyter-execute::

   difference = topo_free_disturbance-bouguer_disturbance

   cpt_lims = vd.minmax(difference, min_percentile=5, max_percentile=95)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="viridis", series=cpt_lims, background=True)
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      fill=difference,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen+ggray'],
   )
   fig.colorbar(
      cmap=True,
      frame=["af", "x+lDifference between Bouguer and Topography-free disturbances", "y+lmGal"],
      position="+e",
   )

   fig.shift_origin(xshift='w+6c')

   cpt_lims = vd.minmax(topography_proj, min_percentile=2, max_percentile=98)

   pygmt.makecpt(cmap="etopo1", series=cpt_lims, background=True)
   fig.grdimage(
      topography,
      cmap=True,
      projection="M15c",
      frame=['ag', 'WSen'],
   )
   fig.colorbar(cmap=True, frame=["af", "x+lTopography", "y+lmeters"])
   fig.show()


Terrain correction in spherical coordinates
-------------------------------------------

So far we computed the terrain effect by projecting the topography grid and
the observation points to plain Cartesian coordinates and approximating the
topographic masses with rectangular prisms.
On regional to global scales the curvature of the Earth cannot be neglected:
the projection distorts the geometry of the topographic masses and the
computed terrain effect accumulates errors.
In such cases we can forward model the topographic masses directly in
geocentric spherical coordinates using tesseroids (spherical prisms), which
take the curvature of the Earth into account.

We can build a model of the topographic masses through the
:func:`harmonica.tesseroid_layer` function.
Unlike :func:`harmonica.prism_layer`, its ``surface`` and ``reference``
arguments must be passed as **radii** measured from the center of the Earth,
not as heights above a reference level.
We can obtain the radii of the surface of the reference ellipsoid at each
latitude with :meth:`boule.Ellipsoid.geocentric_radius` and add the
topographic heights to them:

.. jupyter-execute::

   import boule as bl

   ellipsoid = bl.WGS84

   longitude, latitude = np.meshgrid(topography.longitude, topography.latitude)
   reference = ellipsoid.geocentric_radius(latitude)
   surface = reference + topography.values

We will assign the same densities we used for the layer of prisms and define
the layer of tesseroids:

.. jupyter-execute::

   density = np.where(topography.values >= 0, 2670, 1040 - 2670)

   tesseroids = hm.tesseroid_layer(
       coordinates=(topography.longitude, topography.latitude),
       surface=surface,
       reference=reference,
       properties={"density": density},
   )
   tesseroids

.. note::

   We are using the geodetic latitude of the topography grid as the latitude
   of the tesseroids, which live in geocentric spherical coordinates.
   This assumes the difference between the two latitudes (up to 0.2 degrees)
   has no significant effect on the terrain correction.
   Converting the grid to geocentric spherical coordinates would avoid the
   assumption, but a regular grid in geodetic coordinates is not regular in
   spherical ones, so the topography would have to be regridded first.

The radial coordinate of the observation points must be expressed in the same
way as the boundaries of the layer: as radii from the center of the Earth.
We will compute them the same way we defined the ``surface`` of the layer, by
adding the observation heights to the geocentric radius of the ellipsoid at
each latitude.
This keeps the observation points consistent with the model of the topographic
masses:

.. jupyter-execute::

   radius = ellipsoid.geocentric_radius(data.latitude) + data.height_geometric_m

Tesseroid forward modelling requires every computation point to be located
outside of the tesseroids.
Since our observations were taken on the terrain surface, some of them fall
below the top of the tesseroid that contains them: the tops of the tesseroids
are given by the topography grid, which averages the terrain over each cell,
while the observation heights were measured at each station.
Rather than moving the observation points, we will trust the measured heights
and lower the top of every tesseroid that contains a station below it to the
radius of that station.
A station that sits exactly on the boundary between two tesseroids belongs to
both, so we look up the tesseroids on every side of each station (shifting its
coordinates by far less than their precision):

.. jupyter-execute::

   indices = np.arange(tesseroids.top.size).reshape(tesseroids.top.shape)
   cells = tesseroids.top.copy(data=indices)
   lowest_station = np.full(tesseroids.top.size, np.inf)
   shift = 1e-9
   for shift_longitude in (-shift, shift):
       for shift_latitude in (-shift, shift):
           index = cells.sel(
               longitude=xr.DataArray(data.longitude + shift_longitude),
               latitude=xr.DataArray(data.latitude + shift_latitude),
               method="nearest",
           )
           np.minimum.at(lowest_station, index.values, radius)
   lowest_station = lowest_station.reshape(tesseroids.top.shape)

   surface = np.minimum(surface, lowest_station)
   tesseroids.tesseroid_layer.update_top_bottom(surface, reference)

.. note::

   The same situation arises with the layer of prisms, but the prism forward
   model doesn't require the computation points to be outside of the prisms,
   so it went unnoticed in the previous section: the mass above those
   stations is still part of that model.

Now we can compute the terrain effect through the
:meth:`harmonica.DatasetAccessorTesseroidLayer.gravity` method:

.. jupyter-execute::

   coordinates_sph = (data.longitude, data.latitude, radius)
   terrain_effect_spherical = tesseroids.tesseroid_layer.gravity(
       coordinates_sph, field="g_z"
   )

And obtain a topography-free gravity disturbance that takes the curvature of
the Earth into account:

.. jupyter-execute::

   topo_free_disturbance_spherical = (
       data.gravity_disturbance_mgal - terrain_effect_spherical
   )

   cpt_lims = vd.minmax(topo_free_disturbance_spherical)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="viridis", series=cpt_lims)
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      fill=topo_free_disturbance_spherical,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen+ggray'],
   )
   fig.colorbar(
      cmap=True,
      frame=[
         "a50f25",
         "x+lTopography-free gravity disturbance (tesseroids)",
         "y+lmGal",
      ],
   )
   fig.show()

Compare the terrain effects of prisms and tesseroids
----------------------------------------------------

Even though this region spans only a few degrees, the two models don't agree.
Let's plot the difference between the terrain effects computed with prisms and
with tesseroids:

.. jupyter-execute::

   difference = terrain_effect - terrain_effect_spherical

   cpt_lims = vd.minmax(difference, min_percentile=5, max_percentile=95)

   fig = pygmt.Figure()
   pygmt.makecpt(cmap="viridis", series=cpt_lims)
   fig.plot(
      x=data.longitude,
      y=data.latitude,
      fill=difference,
      cmap=True,
      style="c3p",
      projection="M15c",
      frame=['ag', 'WSen+ggray'],
   )
   fig.colorbar(
      cmap=True,
      frame=["af", "x+lTerrain effect difference (prisms - tesseroids)", "y+lmGal"],
   )
   fig.show()

The tesseroids produce a terrain effect that is systematically larger, by
about 4.5 mGal on average, than the one produced by the prisms.
This is the effect of the curvature of the Earth on the terrain correction: the
topographic masses far from an observation point lie below the plane that is
tangent to the Earth at that point, so they pull more strongly downwards than
the same masses laid flat in a Cartesian model.
Unlike the difference between the Bouguer and the topography-free
disturbances, this one is fairly uniform: it depends on how much topography
surrounds each station rather than on how rugged it is, so it grows with the
extent of the topography grid and shrinks towards its edges.
For this grid, which extends a few hundred kilometers around the observations,
it amounts to 3-4% of the terrain effect.
Whether that is negligible depends on the goal of the survey: it's comparable
to the differences we found above between the Bouguer and the topography-free
disturbances.

.. hint::

   This is the same effect that the classic Bullard B (curvature) correction
   accounts for when applying a Bouguer correction with a spherical cap
   instead of an infinite slab.

The largest differences are found at stations that lie below the topography
grid. The prism model still has topographic mass above them, which pulls
upwards, while we removed it from the tesseroid model.

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <topographic_correction>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <topographic_correction>`
        :text-align: center
