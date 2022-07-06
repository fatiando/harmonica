.. _equivalent_sources:

Equivalent Sources
==================

.. toctree::
   :hidden:

   eqs-parameters-estimation
   block-averaged-eqs
   gradient-boosted-eqs
   eq-sources-spherical

Most potential field surveys gather data along scattered and uneven flight
lines or ground measurements. For a great number of applications we may need to
interpolate these data points onto a regular grid at a constant altitude.
Upward-continuation is also a routine task for smoothing, noise attenuation,
source separation, etc.

Both tasks can be done simultaneously through an *equivalent sources*
[Dampney1969]_ (a.k.a *equivalent layer*). The equivalent sources technique
consists in defining a finite set of geometric bodies (like point sources)
beneath the observation points and adjust their coefficients so they generate
the same measured field on the observation points. These fitted sources can be
then used to *predict* the values of this field on any unobserved location,
like a regular grid, points at different heights, any set of scattered
points or even along a profile.

The equivalent sources have two major advantages over any general purpose
interpolator:

* it takes into account the 3D nature of the potential fields being measured by
  considering the observation heights, and
* its predictions are always harmonic.

Its main drawback is the increased computational load it takes to fit the
sources' coefficients, both in terms of memory and computation time).

Harmonica has a few different classes for applying the equivalent sources
techniques. Here we will explore how we can use the
:class:`harmonica.EquivalentSources` to interpolate some gravity disturbance
scattered points on a regular grid with a small upward continuation.

We can start by downloading some sample gravity data over the Bushveld Igneous
Complex in Southern Africa:

.. jupyter-execute::

   import ensaio
   import pandas as pd

   fname = ensaio.fetch_bushveld_gravity(version=1)
   data = pd.read_csv(fname)
   data

The :class:`harmonica.EquivalentSources` class works exclusively in Cartesian
coordinates, so we need to project these gravity observations:

.. jupyter-execute::

   import pyproj

   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.values.mean())
   easting, northing = projection(data.longitude.values, data.latitude.values)

Now we can initialize the :class:`harmonica.EquivalentSources` class.


.. jupyter-execute::

   import harmonica as hm

   equivalent_sources = hm.EquivalentSources(depth=10e3, damping=10)
   equivalent_sources

By default, it places the sources one beneath each data point at a relative
depth from the elevation of the data point following [Cooper2000]_.
This *relative depth* can be set through the ``depth`` argument.
Deepest sources generate smoother predictions (*underfitting*), while shallow
ones tend to overfit the data.

.. note::

   If instead we want to place every source at a constant depth, we can change
   it by passing  ``depth_type="constant"``. In that case, the ``depth``
   argument will be the exact depth at which the sources will be located.

The ``damping`` parameter is used to smooth the coefficients of the sources and
stabilize the least square problem. A higher ``damping`` will create smoother
predictions, while a lower one could overfit the data and create artifacts.

Now we can estimate the source coefficients through the
:meth:`harmonica.EquivalentSources.fit` method against the observed gravity
disturbance.

.. jupyter-execute::

   coordinates = (easting, northing, data.height_geometric_m)
   equivalent_sources.fit(coordinates, data.gravity_disturbance_mgal)

Once the fitting process finishes, we can predict the values of the field on
any set of points using the :meth:`harmonica.EquivalentSources.predict` method.
For example, lets predict on the same observation points to check if the
sources are able to reproduce the observed field.

.. jupyter-execute::

   disturbance = equivalent_sources.predict(coordinates)


And plot it:

.. jupyter-execute::

   import verde as vd
   import matplotlib.pyplot as plt

   # Get max absolute value for the observed gravity disturbance
   maxabs = vd.maxabs(data.gravity_disturbance_mgal)

   # Create a figure with two axes
   fig, (ax1, ax2) = plt.subplots(figsize=(10, 12), nrows=2, ncols=1, sharex=True)

   # Plot observed and predicted fields
   tmp = ax1.scatter(
       easting,
       northing,
       c=data.gravity_disturbance_mgal,
       s=2,
       vmin=-maxabs,
       vmax=maxabs,
       cmap="seismic",
   )
   ax1.set_title("Observed gravity disturbance")
   ax2.scatter(
       easting,
       northing,
       c=disturbance,
       s=2,
       vmin=-maxabs,
       vmax=maxabs,
       cmap="seismic",
   )
   ax2.set_title("Predicted gravity disturbance")

   for ax in (ax1, ax2):
       ax.set_aspect("equal")
       plt.colorbar(tmp, ax=ax, label="mGal", pad=0.05, aspect=20, shrink=0.9)
   plt.show()

We can also *grid* and *upper continue* the field by predicting its values on
a regular grid at a constant height higher than the observations. To do so we
can use the :func:`verde.grid_coordinates` function to create the coordinates
of the grid and then use the :meth:`harmonica.EquivalentSources.grid` method.

First, lets get the maximum height of the observations:

.. jupyter-execute::

   data.height_geometric_m.max()

Then create the grid coordinates at a constant height of  and a spacing of 2km;
and use the equivalent sources to generate a gravity disturbance grid.

.. jupyter-execute::

   # Get region of the observations
   region = vd.get_region(coordinates)

   # Build the grid coordinates
   grid_coords = vd.grid_coordinates(region=region, spacing=2e3, extra_coords=2.2e3)

   # Grid the gravity disturbances
   grid = equivalent_sources.grid(grid_coords, data_names=["gravity_disturbance"])
   grid

And plot it

.. jupyter-execute::

   fig = plt.figure(figsize=(12, 8))

   grid.gravity_disturbance.plot()
   plt.gca().set_aspect("equal")
   plt.show()
