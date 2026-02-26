.. _prism:

Rectangular prisms
==================

One of the geometric bodies that Harmonica is able to forward model are
*rectangular prisms*.
We can compute their gravitational fields through the
:func:`harmonica.prism_gravity` function.

Rectangular prisms can only be defined in Cartesian coordinates, therefore they
are intended to be used to forward model bodies in a small geographic region,
in which we can neglect the curvature of the Earth.

Each prism can be defined as a tuple containing its boundaries in the following
order: *west*, *east*, *south*, *north*, *bottom*, *top* in Cartesian
coordinates and in meters.

.. jupyter-execute::
   :hide-code:

   import harmonica as hm

Let's define a single prism and compute the gravitational potential
it generates on a computation point located at 500 meters above its uppermost
surface.

Define the prism and its density (in kg per cubic meters):

.. jupyter-execute::

   prism = (-100, 100, -100, 100, -1000, 500)
   density = 3000

Define a computation point located 500 meters above the *top* surface of the
prism:

.. jupyter-execute::

   coordinates = (0, 0, 1000)

And finally, compute the gravitational potential field it generates on the
computation point:

.. jupyter-execute::

   potential = hm.prism_gravity(coordinates, prism, density, field="potential")
   print(potential, "J/kg")


Gravitational fields
--------------------

The :func:`harmonica.prism_gravity` is able to compute the gravitational
potential (``"potential"``), the acceleration components (``"g_e"``, ``"g_n"``,
``"g_z"``), and tensor components (``"g_ee"``, ``"g_nn"``, ``"g_zz"``,
``"g_en"``, ``"g_ez"``, ``"g_nz"``).


Build a regular grid of computation points located 10m above the zero height:

.. jupyter-execute::

   import verde as vd

   region = (-10e3, 10e3, -10e3, 10e3)
   shape = (51, 51)
   height = 10
   coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)

Define a single prism:

.. jupyter-execute::

   prism = [-2e3, 2e3, -2e3, 2e3, -1.6e3, -900]
   density = 3300

Compute the gravitational fields that this prism generate on each observation
point:

.. jupyter-execute::

   fields = (
      "potential",
      "g_e", "g_n", "g_z",
      "g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"
   )

   results = {}
   for field in fields:
      results[field] = hm.prism_gravity(coordinates, prism, density, field=field)

We can reshape the results into variables of an dataset:

.. jupyter-execute::

   grid = vd.make_xarray_grid(
      coordinates,
      tuple(results.values()),
      data_names=results.keys(),
      extra_coords_names="extra",
   )
   print(grid)

Plot the results:

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

   fig = pygmt.Figure()

   fig.grdimage(
      projection="X10c",
      grid=grid.potential,
      frame=["a", "x+leasting (m)", "y+lnorthing (m)"],
      cmap="viridis",
   )

   fig.colorbar(cmap=True, position="JMR", frame=["x+lJ kg@+-1@+"])
   fig.show()


.. jupyter-execute::

   fig = pygmt.Figure()

   maxabs = vd.maxabs(grid.g_e)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_e,
      projection="X10c",
      cmap=True,
      frame=["WSne+tEasting component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.g_n)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_n,
      projection="X10c",
      cmap=True,
      frame=["wSne+tNorthing component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.g_z)
   pygmt.makecpt(cmap="balance+h0", series=[0, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_z,
      projection="X10c",
      cmap=True,
      frame=["wSnE+tDownward component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.show()

.. jupyter-execute::

   fig = pygmt.Figure()

   maxabs = vd.maxabs(grid.g_ee)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_ee,
      projection="X10c",
      cmap=True,
      frame=["WSne+tEasting-easting tensor", "xa","ya"],
   )
   fig.colorbar(frame='+lEotvos')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.g_nn)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_nn,
      projection="X10c",
      cmap=True,
      frame=["wSne+tNorthing-northing tensor", "xa","ya"],
   )
   fig.colorbar(frame='+lEotvos')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.g_zz)
   pygmt.makecpt(cmap="balance+h0", series=[0, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_zz,
      projection="X10c",
      cmap=True,
      frame=["wSnE+tDownward-downward tensor", "xa","ya"],
   )
   fig.colorbar(frame='+lEotvos')

   fig.show()

.. jupyter-execute::

   fig = pygmt.Figure()

   maxabs = vd.maxabs(grid.g_en)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_en,
      projection="X10c",
      cmap=True,
      frame=["WSne+tEasting-northing tensor", "xa","ya"],
   )
   fig.colorbar(frame='+lEotvos')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.g_ez)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_ez,
      projection="X10c",
      cmap=True,
      frame=["wSne+tEasting-downward tensor", "xa","ya"],
   )
   fig.colorbar(frame='+lEotvos')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.g_nz)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.g_nz,
      projection="X10c",
      cmap=True,
      frame=["wSnE+tNorthing-downward tensor", "xa","ya"],
   )
   fig.colorbar(frame='+lEotvos')

   fig.show()


Passing multiple prisms
~~~~~~~~~~~~~~~~~~~~~~~

We can compute the gravitational field of a set of prisms by passing a list of
them, where each prism is defined as mentioned before, and then making
a single call of the :func:`harmonica.prism_gravity` function.

Lets define a set of four prisms, along with their densities:

.. jupyter-execute::

   prisms = [
       [2e3, 3e3, 2e3, 3e3, -10e3, -1e3],
       [3e3, 4e3, 7e3, 8e3, -9e3, -1e3],
       [7e3, 8e3, 1e3, 2e3, -7e3, -1e3],
       [8e3, 9e3, 6e3, 7e3, -8e3, -1e3],
   ]
   densities = [2670, 3300, 2900, 2980]

We can define a set of computation points located on a regular grid at zero
height:

.. jupyter-execute::

   import verde as vd

   coordinates = vd.grid_coordinates(
       region=(0, 10e3, 0, 10e3), shape=(40, 40), extra_coords=0
   )

And finally calculate the vertical component of the gravitational acceleration
generated by the whole set of prisms on every computation point:

.. jupyter-execute::

   g_z = hm.prism_gravity(coordinates, prisms, densities, field="g_z")

.. note::

   When passing multiple prisms and coordinates to
   :func:`harmonica.prism_gravity` we calculate the field in parallel using
   multiple CPUs, speeding up the computation.

Lets plot this gravitational field:

.. jupyter-execute::

   grid = vd.make_xarray_grid(
      coordinates, g_z, data_names="g_z", extra_coords_names="extra")
   fig = pygmt.Figure()
   fig.grdimage(
      region=(0, 10e3, 0, 10e3),
      projection="X10c",
      grid=grid.g_z,
      frame=["WSne", "x+leasting (m)", "y+lnorthing (m)"],
      cmap='viridis',)
   fig.colorbar(cmap=True, position="JMR", frame=["a2", "x+lmGal"])
   fig.show()


Magnetic fields
---------------

The :func:`harmonica.prism_magnetic` function allows us to calculate the
magnetic fields generated by rectangular prisms on a set of observation points.
Each rectangular prism is defined in the same way we did for the gravity
forward modelling, although we now need to define a magnetization vector for
each one of them.

For example:

.. jupyter-execute::

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

The ``magnetization`` should be a tuple of three arrays: the easting, northing
and upward components of the magnetization vector (in :math:`Am^{-1}`) for each
prism in ``prisms``.

We can use the :func:`harmonica.prism_magnetic` function to compute the three
components of the magnetic field the prisms generate on any set of observation
points by choosing ``field="b"``:

.. jupyter-execute::

   region = (-10e3, 10e3, -10e3, 10e3)
   shape = (51, 51)
   height = 10
   coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)

.. jupyter-execute::

   b_e, b_n, b_u = hm.prism_magnetic(coordinates, prisms, magnetization, field="b")


We can reshape the results into variables of an dataset:

.. jupyter-execute::

   grid = vd.make_xarray_grid(
      coordinates,
      data=(b_e, b_n, b_u),
      data_names=["b_e", "b_n", "b_u"],
      extra_coords_names="extra"
   )

.. jupyter-execute::

   fig = pygmt.Figure()

   maxabs = vd.maxabs(grid.b_e)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.b_e,
      projection="X10c",
      cmap=True,
      frame=["WSne+tEasting component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.b_n)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.b_n,
      projection="X10c",
      cmap=True,
      frame=["wSne+tNorthing component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.shift_origin(xshift="11c")

   maxabs = vd.maxabs(grid.b_u)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.b_u,
      projection="X10c",
      cmap=True,
      frame=["wSnE+tUpward component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.show()


Alternatively, we can compute just a single component by choosing ``field`` to
be:

* ``"b_e"`` for the easting component,
* ``"b_n"`` for the northing component, and
* ``"b_u"`` for the upward component.

.. important::

   Computing the three components at the same time with ``field="b"`` is more
   efficient than computing each one of the three components independently.

For example, we can calculate only the upward component of the magnetic field
generated by these two prisms:

.. jupyter-execute::

   b_u = hm.prism_magnetic(
      coordinates, prisms, magnetization, field="b_u"
   )

.. jupyter-execute::

   fig = pygmt.Figure()

   grid = vd.make_xarray_grid(
      coordinates, b_u, data_names="b_u", extra_coords_names="extra")

   maxabs = vd.maxabs(grid.b_u)
   pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)
   fig.grdimage(
      grid=grid.b_u,
      projection="X10c",
      cmap=True,
      frame=["WSne+tUpward component", "xa","ya"],
   )
   fig.colorbar(frame='+lnT')

   fig.show()


.. _prism_layer:

Prism layer
-----------

One of the most common usage of prisms is to model geologic structures.
Harmonica offers the possibility to define a layer of prisms through the
:func:`harmonica.prism_layer` function: a regular grid of
prisms of equal size along the horizontal dimensions and with variable top and
bottom boundaries.
It returns a :class:`xarray.Dataset` with the coordinates of the centers of the
prisms and their corresponding physical properties.

The :class:`harmonica.DatasetAccessorPrismLayer` Dataset accessor can be used
to obtain some properties of the layer like its shape and size or the
boundaries of any prism in the layer.
Moreover, we can use the :meth:`harmonica.DatasetAcessorPrismLayer.gravity`
method to compute the gravitational field of the prism layer on any set of
computation points.

Lets create a simple prism layer, whose lowermost boundaries will be set on
zero and their uppermost boundary will approximate a sinusoidal function.
We can start by setting the region of the layer and the horizontal dimensions
of the prisms:

.. jupyter-execute::

   region = (0, 100e3, -40e3, 40e3)
   spacing = 2000

Then we can define a regular grid where the centers of the prisms will fall:

.. jupyter-execute::

   easting, northing = vd.grid_coordinates(region=region, spacing=spacing)

We need to define a 2D array for the uppermost *surface* of the layer. We will
sample a trigonometric function for this simple example:

.. jupyter-execute::

   wavelength = 24 * spacing
   surface = np.abs(np.sin(easting * 2 * np.pi / wavelength))

Let's assign the same density to each prism through a 2d array with the same
value: 2700 kg per cubic meter.

.. jupyter-execute::

   density = np.full_like(surface, 2700)

Now we can define the prism layer specifying the reference level to zero:

.. jupyter-execute::

   prisms = hm.prism_layer(
       coordinates=(easting, northing),
       surface=surface,
       reference=0,
       properties={"density": density},
   )

Let's define a grid of observation points at 1 km above the zeroth height:

.. jupyter-execute::

   region_pad = vd.pad_region(region, 10e3)
   coordinates = vd.grid_coordinates(
       region_pad, spacing=spacing, extra_coords=1e3
   )


And compute the gravitational field generated by the prism layer on them:

.. jupyter-execute::

   gravity = prisms.prism_layer.gravity(coordinates, field="g_z")

Finally, let's plot the gravitational field:

.. jupyter-execute::

   grid = vd.make_xarray_grid(
      coordinates, gravity, data_names="gravity", extra_coords_names="extra")

   fig = pygmt.Figure()
   title = "Gravitational acceleration of a layer of prisms"
   fig.grdimage(
      region=region_pad,
      projection="X10c",
      grid=grid.gravity,
      frame=[f"WSne+t{title}", "x+leasting (m)", "y+lnorthing (m)"],
      cmap='viridis',)
   fig.colorbar(cmap=True, position="JMR", frame=["a.02", "x+lmGal"])
   fig.show()

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <prism>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <prism>`
        :text-align: center
