.. _eq-sources-spherical:

Equivalent sources in spherical coordinates
===========================================

When interpolating gravity or magnetic data over a very large region that
covers full continents we need to take into account the curvature of the Earth.
In these cases projecting the data to plain Cartesian coordinates may introduce
errors due to the distorsions caused by it.
Therefore using the :class:`harmonica.EquivalentSources` class is not well
suited for it.

Instead, we can make use of equivalent sources that are defined in
:ref:`spherical geocentric coordinates <spherical_coordinates>` through the
:class:`harmonica.EquivalentSourcesSph` class.

Lets start by fetching some gravity data over Southern Africa:

.. jupyter-execute::

    import ensaio
    import pandas as pd

    fname = ensaio.fetch_southern_africa_gravity(version=1)
    data = pd.read_csv(fname)
    data

To speed up the computations on this simple example we are going to downsample
the data using a blocked mean to a resolution of 6 arcminutes.

.. jupyter-execute::

    import numpy as np
    import verde as vd

    blocked_mean = vd.BlockReduce(np.mean, spacing=6 / 60, drop_coords=False)
    (longitude, latitude, height), gravity_data = blocked_mean.filter(
        (data.longitude, data.latitude, data.height_sea_level_m),
        data.gravity_mgal,
    )

    data = pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "height_sea_level_m": height,
            "gravity_mgal": gravity_data,
        }
    )
    data

And compute gravity disturbance by subtracting normal gravity using
:mod:`boule` (see :ref:`gravity_disturbance` for more information):

.. jupyter-execute::

    import boule as bl

    ellipsoid = bl.WGS84
    normal_gravity = ellipsoid.normal_gravity(data.latitude, data.height_sea_level_m)
    gravity_disturbance = data.gravity_mgal - normal_gravity

Lets define some equivalent sources in spherical coordinates. We will choose
some first guess values for the ``damping`` and ``depth`` parameters.

.. jupyter-execute::

    import harmonica as hm

    eqs = hm.EquivalentSourcesSph(damping=1e-3, relative_depth=10000)

.. seealso::

    Check how we can :ref:`estimate the damping and depth parameters
    <eqs-parameters-estimation>` using a cross-validation.

Before we can fit the sources' coefficients we need to convert the data given
in geographical coordinates to spherical ones. We can do it through the
:meth:`boule.Ellipsoid.geodetic_to_spherical` method of the WGS84 ellipsoid
defined in :mod:`boule`.

.. jupyter-execute::

    coordinates = ellipsoid.geodetic_to_spherical(
        data.longitude, data.latitude, data.height_sea_level_m
    )

And then use them to fit the sources:

.. jupyter-execute::

   eqs.fit(coordinates, gravity_disturbance)

We can then use these sources to predict the gravity disturbance on a regular
grid defined in geodetic coordinates. We will generate a regular grid of
computation points located at the maximum height of the data and with a spacing
of 6 arcminutes.

.. jupyter-execute::

    # Get the bounding region of the data in geodetic coordinates
    region = vd.get_region((data.longitude, data.latitude))

    # Get the maximum height of the data coordinates
    max_height = data.height_sea_level_m.max()

    # Define a regular grid of points in geodetic coordinates
    grid_coords = vd.grid_coordinates(
        region=region, spacing=6 / 60, extra_coords=max_height
    )

But before we can tell the equivalent sources to predict the
field we need to convert the grid coordinates to spherical.

.. jupyter-execute::

    grid_coords_sph = ellipsoid.geodetic_to_spherical(*grid_coords)

And then predict the gravity disturbance on the grid points:

.. jupyter-execute::

    gridded_disturbance = eqs.predict(grid_coords_sph)

Lastly we can generate a :class:`xarray.DataArray` using
:func:`verde.make_xarray_grid`:

.. jupyter-execute::

    grid = vd.make_xarray_grid(
        grid_coords,
        gridded_disturbance,
        data_names=["gravity_disturbance"],
        extra_coords_names="upward",
    )
    grid

Since the data points don't cover the entire area, we might want to mask those
grid points that are too far away from any data point:

.. jupyter-execute::

    grid = vd.distance_mask(
        data_coordinates=(data.longitude, data.latitude), maxdist=0.5, grid=grid
    )

Lets plot it:

.. jupyter-execute::

    import pygmt

    maxabs = vd.maxabs(gravity_disturbance, grid.gravity_disturbance.values)

    fig = pygmt.Figure()

    # Make colormap of data
    pygmt.makecpt(cmap="polar+h0",series=(-maxabs, maxabs,))

    title = "Block-median reduced gravity disturbance"
    fig.plot(
        projection="M100/15c",
        region=region,
        frame=[f"WSne+t{title}", "xa5", "ya4"],
        x=longitude,
        y=latitude,
        color=gravity_disturbance,
        style="c0.1c",
        cmap=True,
    )
    fig.coast(shorelines="0.5p,black", area_thresh=1e4)
    fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

    fig.shift_origin(xshift='w+3c')

    title = "Gridded gravity disturbance"
    fig.grdimage(
        grid=grid.gravity_disturbance,
        cmap=True,
        frame=[f"ESnw+t{title}","xa5", "ya4"],
        nan_transparent=True,
    )
    fig.coast(shorelines="0.5p,black", area_thresh=1e4)
    fig.colorbar(cmap=True, frame=["a50f25", "x+lmGal"])

    fig.show()

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <eq-sources-spherical>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <eq-sources-spherical>`
        :text-align: center
