.. _tesseroid:

Tesseroids
==========

When our region of interest covers several longitude and latitude degrees,
utilizing Cartesian coordinates to model geological structures might
introduce significant errors: they don't take into account the curvature of the
Earth. Instead, we would need to work in :ref:`spherical_coordinates`.
A common approach to forward model bodies in geocentric spherical coordinates
is to make use of tesseroids.

A tesseroid (a.k.a spherical prism) is a three dimensional body defined by the
volume contained by two longitudinal boundaries, two latitudinal boundaries and
the surfaces of two concentric spheres of different radii (see :ref:`tesseroid
figure`).

.. figure:: ../../_static/figures/tesseroid.svg
   :name: tesseroid figure
   :width: 50%
   :alt: Figure showing a tesseroid defined in a geocentric spherical coordinate system

   Figure: Tesseroid

   Tesseroid defined by two longitude coordinates (:math:`\lambda_1` and
   :math:`\lambda_2`), two latitude coordinates (:math:`\phi_1` and
   :math:`\phi_2`) and the surfaces of two concentric spheres of radii
   :math:`r_1` and :math:`r_2`.
   This figure is a modified version of [Uieda2015]_.


Through the :func:`harmonica.tesseroid_gravity` function we can calculate the
gravitational field of any tesseroid with a given density on any computation
point. Each tesseroid can be represented through a tuple containing its six
boundaries in the following order: *west*, *east*, *south*, *north*, *bottom*,
*top*, where the former four are its longitudinal and latitudinal boundaries in
decimal degrees and the latter two are the two radii given in meters.

These two radii represent the top and bottom surfaces of the tesseroid, and should be
given as distances from the center of the Earth. Note this is different from the
vertical boundaries used for **prisms** in Cartesian coordinates, which are given as
heights above or below some reference level (e.g., mean sea level or a reference ellipsoid).

.. note::

   The :func:`harmonica.tesseroid_gravity` numerically computed the
   gravitational fields of tesseroids by applying a method that applies the
   Gauss-Legendre Quadrature along with a bidimensional adaptive discretization
   algorithm. Refer to [Soler2019]_ for more details.


Lets define a single tesseroid and compute the gravitational potential
it generates on a regular grid of computation points located at 10 km above
its *top* boundary.

Get the WGS84 reference ellipsoid from :mod:`boule` so we can obtain its mean
radius:

.. jupyter-execute::

   import boule as bl

   ellipsoid = bl.WGS84
   mean_radius = ellipsoid.mean_radius

Define the tesseroid and its density (in kg per cubic meters):

.. jupyter-execute::

   tesseroid = (-70, -50, -40, -20, mean_radius - 10e3, mean_radius)
   density = 2670

Define a set of computation points located on a regular grid at 100 km above
the *top* surface of the tesseroid:

.. jupyter-execute::

   import bordado as bd

   coordinates = bd.grid_coordinates(
       region=[-80, -40, -50, -10],
       shape=(80, 80),
       non_dimensional_coords=100e3 + mean_radius,
   )

Lets compute the *downward* component of the gravitational acceleration it
generates on the computation point:

.. jupyter-execute::

   import harmonica as hm

   gravity = hm.tesseroid_gravity(coordinates, tesseroid, density, field="g_z")

.. important::

   The *downward* component :math:`g_z` of the gravitational acceleration
   computed in spherical coordinates corresponds to :math:`-g_r`, where
   :math:`g_r` is the
   radial component.

And finally plot the computed gravitational field

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


Multiple tesseroids
-------------------

We can compute the gravitational field of a set of tesseroids by passing a list
of them, where each tesseroid is defined as mentioned before, and then making
a single call of the :func:`harmonica.tesseroid_gravity` function.

Lets define a set of four prisms along with their densities:

.. jupyter-execute::

   tesseroids = [
       [-70, -65, -40, -35, mean_radius - 100e3, mean_radius],
       [-55, -50, -40, -35, mean_radius - 100e3, mean_radius],
       [-70, -65, -25, -20, mean_radius - 100e3, mean_radius],
       [-55, -50, -25, -20, mean_radius - 100e3, mean_radius],
   ]
   densities = [2670 , 2670, 2670, 2670]

Compute their gravitational effect on a grid of computation points:

.. jupyter-execute::

   coordinates = bd.grid_coordinates(
       region=[-80, -40, -50, -10],
       shape=(80, 80),
       non_dimensional_coords=100e3 + mean_radius,
   )
   gravity = hm.tesseroid_gravity(coordinates, tesseroids, densities, field="g_z")

And plot the results:

.. jupyter-execute::

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


Tesseroids with variable density
--------------------------------

The :func:`harmonica.tesseroid_gravity` is capable of computing the
gravitational effects of tesseroids whose density is defined through
a continuous function of the radial coordinate. This is achieved by the
application of the method introduced in [Soler2021]_.

To do so we need to define a regular Python function for the density, which
should have a single argument (the ``radius`` coordinate) and return the
density of the tesseroids at that radial coordinate.
In addition, we need to decorate the density function with
:func:`numba.jit(nopython=True)` or ``numba.njit`` for short.

Lets compute the gravitational effect of four tesseroids whose densities are
given by a custom linear ``density`` function.

Start by defining the tesseroids

.. jupyter-execute::

   tesseroids = (
       [-70, -60, -40, -30, mean_radius - 3e3, mean_radius],
       [-70, -60, -30, -20, mean_radius - 5e3, mean_radius],
       [-60, -50, -40, -30, mean_radius - 7e3, mean_radius],
       [-60, -50, -30, -20, mean_radius - 10e3, mean_radius],
   )

Then, define a linear density function. We need to use the ``jit`` decorator so
Numba can run the forward model efficiently.

.. jupyter-execute::

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

Lets create a set of computation points located on a regular grid at 100km
above the mean Earth radius:

.. jupyter-execute::

   coordinates = bd.grid_coordinates(
       region=[-80, -40, -50, -10],
       shape=(80, 80),
       non_dimensional_coords=100e3 + ellipsoid.mean_radius,
   )

And compute the gravitational fields the tesseroids generate:

.. jupyter-execute::

   gravity = hm.tesseroid_gravity(coordinates, tesseroids, density, field="g_z")

Finally, lets plot it:

.. jupyter-execute::

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


.. _tesseroid_layer:

Tesseroid layer
---------------

A common use of tesseroids is to model geologic structures on regional or global
scales, where the curvature of the Earth cannot be neglected. Harmonica offers the
possibility to define a layer of tesseroids through the
:func:`harmonica.tesseroid_layer` function: a regular grid of tesseroids of equal size
along the longitudinal and latitudinal dimensions and with variable top and bottom
boundaries.
It returns a :class:`xarray.Dataset` with the coordinates of the centers of the
tesseroids and their corresponding physical properties.

The :class:`harmonica.DatasetAccessorTesseroidLayer` Dataset accessor can be used to
obtain some properties of the layer like its shape and size or the boundaries of any
tesseroid in the layer.
Moreover, we can use the :meth:`harmonica.DatasetAccessorTesseroidLayer.gravity` method
to compute the gravitational field of the tesseroid layer on any set of computation
points.

.. important::

   Unlike the :func:`harmonica.prism_layer`, the ``surface`` and ``reference`` boundaries
   of a tesseroid layer must be given as **radii** measured from the center of the Earth,
   not as heights above a reference level.
   We can use the :meth:`boule.geocentric_radius` method from :mod:`boule` to
   obtain the radius of the reference ellipsoid at each latitude, and add our height values
   to it.

Let's create a simple tesseroid layer over a region in South America, whose top boundary
will approximate a synthetic topography and whose bottom boundary will be set on the
surface of the reference ellipsoid.
We can start by getting the WGS84 reference ellipsoid from :mod:`boule` and defining the
region of the layer and the horizontal dimensions of the tesseroids (in degrees):

.. jupyter-execute::

   import boule as bl

   ellipsoid = bl.WGS84
   region = (-80, -40, -50, -10)
   spacing = 0.5

Then we can define a regular grid where the centers of the tesseroids will fall:

.. jupyter-execute::

   import bordado as bd

   longitude, latitude = bd.grid_coordinates(region=region, spacing=spacing)

The bottom boundary of the layer (``reference``) will be the surface of the ellipsoid,
so we compute its geocentric radius at each latitude:

.. jupyter-execute::

   reference = ellipsoid.geocentric_radius(latitude)

We need to define a 2D array with the radii of the uppermost *surface* of the layer. We
will build a synthetic topography and add it to the reference radii so that the ``surface``
is also expressed as radii from the center of the Earth:

.. jupyter-execute::

   import numpy as np

   topography = 3e3 * np.sin(longitude * np.pi / 20) * np.cos(latitude * np.pi / 20)
   surface = reference + topography

Let's assign the same density to each tesseroid through a 2D array with the same value:
2670 kg per cubic meter.

.. jupyter-execute::

   density = np.full_like(surface, 2670.0)

Now we can define the tesseroid layer:

.. jupyter-execute::

   import harmonica as hm

   tesseroids = hm.tesseroid_layer(
       coordinates=(longitude, latitude),
       surface=surface,
       reference=reference,
       properties={"density": density},
   )
   tesseroids

Let's define a grid of observation points located 10 km above the reference ellipsoid.
Since the radius of the ellipsoid changes with latitude, we compute the radial coordinate
of the observation points accordingly:

.. jupyter-execute::

   grid_longitude, grid_latitude = bd.grid_coordinates(region=region, spacing=spacing)
   grid_radius = ellipsoid.geocentric_radius(grid_latitude) + 10e3
   coordinates = (grid_longitude, grid_latitude, grid_radius)

And compute the *downward* component of the gravitational acceleration generated by the
tesseroid layer on them:

.. jupyter-execute::

   gravity = tesseroids.tesseroid_layer.gravity(coordinates, field="g_z")

Finally, let's plot the gravitational field:

.. jupyter-execute::

   import verde as vd

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

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <tesseroid>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <tesseroid>`
        :text-align: center
