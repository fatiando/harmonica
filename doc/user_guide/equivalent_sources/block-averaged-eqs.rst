.. _block-averaged-eqs:

Block-averaged equivalent sources
=================================

When we introduced the :ref:`equivalent_sources` we saw that
by default the :class:`harmonica.EquivalentSources` class locates one point
source beneath each data point during the fitting process, following
[Cooper2000]_.

Alternatively, we can use another strategy: the *block-averaged sources*,
introduced in [Soler2021]_.
This method divides the survey region (defined by the data) into square blocks
of equal size, computes the median coordinates of the data points that fall
inside each block and locates one source beneath every averaged position. This
way, we define one equivalent source per block, with the exception of empty
blocks that won't get any source.

This method has two main benefits:

- It lowers the amount of sources involved in the interpolation, therefore it
  reduces the computer memory requirements and the computation time of the
  fitting and prediction processes.
- It might avoid to produce aliasing on the output grids, specially for
  surveys with oversampling along a particular direction, like airborne ones.

We can make use of the block-averaged sources within the
:class:`harmonica.EquivalentSources` class by passing a value to the
``block_size`` parameter, which controls the size of the blocks.

Lets load some total-field magnetic data over Great Britain:

.. jupyter-execute::

    import ensaio
    import pandas as pd

    fname = ensaio.fetch_britain_magnetic(version=1)
    data = pd.read_csv(fname)
    data

In order to speed-up calculations we are going to slice a smaller portion of
the data:

.. jupyter-execute::

    import verde as vd

    region = (-5.5, -4.7, 57.8, 58.5)
    inside = vd.inside((data.longitude, data.latitude), region)
    data = data[inside]
    data

And project the geographic coordinates to plain Cartesian ones:

.. jupyter-execute::

    import pyproj

    projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
    easting, northing = projection(data.longitude.values, data.latitude.values)
    coordinates = (easting, northing, data.height_m)
    xy_region=vd.get_region(coordinates)

.. jupyter-execute::

    import pygmt

    maxabs = vd.maxabs(data.total_field_anomaly_nt)*.8

    # Set figure properties
    w, e, s, n = xy_region
    fig_height = 15
    fig_width = fig_height * (e - w) / (n - s)
    fig_ratio = (n - s) / (fig_height / 100)
    fig_proj = f"x1:{fig_ratio}"

    # Plot original magnetic anomaly and the gridded and upward-continued version
    fig = pygmt.Figure()

    title = "Observed total-field magnetic anomaly"

    pygmt.makecpt(
        cmap="polar+h0",
        series=(-maxabs, maxabs),
        background=True,
    )

    with pygmt.config(FONT_TITLE="12p"):
        fig.plot(
            projection=fig_proj,
            region=xy_region,
            frame=[f"WSne+t{title}", "xa10000", "ya10000"],
            x=easting,
            y=northing,
            color=data.total_field_anomaly_nt,
            style="c0.1c",
            cmap=True,
        )
    fig.colorbar(cmap=True, position="JMR", frame=["a200f100", "x+lnT"])
    fig.show()


Most airborne surveys like this one present an anysotropic distribution of the
data: there are more observation points along the flight lines that goes west
to east than the ones going south to north.
Placing a single source beneath each observation point generates an anysotropic
distribution of the equivalent sources, which might lead to aliases on the
generated outputs.

Instead, we can use the **block-averaged equivalent sources** by
creating a :class:`harmonica.EquivalentSources` instance passing the size of
the blocks through the ``block_size`` parameter.

.. jupyter-execute::

    import harmonica as hm

    eqs = hm.EquivalentSources(
        depth=1000, damping=1, block_size=500, depth_type="constant"
    )

These sources were set at a constant depth of 1km bellow the zeroth height and
with a ``damping`` equal to 1. See how you can choose values for these
parameters in :ref:`eqs-parameters-estimation`.

.. note::

    The depth of the sources can be set analogously to the regular equivalent
    sources: we can use a ``constant`` depth (every source is located at the same
    depth) or a ``relative`` depth (where each source is located at a constant
    shift beneath the median location obtained during the block-averaging process).
    The depth of the sources and which strategy to use can be set up through the
    ``depth`` and the ``depth_type`` parameters, respectively.

.. important::

    We recommend using a ``block_size`` not larger than the desired resolution
    of the interpolation grid.

Now we can fit the equivalent sources against the magnetic data. During this
step the point sources are created through the block averaging process.

.. jupyter-execute::

    eqs.fit(coordinates, data.total_field_anomaly_nt)

.. tip::

    We can obtain the coordinates of the created sources through the ``points_``
    attribute. Lets see how many sources it created:

    .. jupyter-execute::

        eqs.points_[0].size

    We have less sources than observation points indeed.


We can finally grid the magnetic data using the block-averaged equivalent
sources. We will generate a regular grid with a resolution of 500 m and at 1500
m height. Since the maximum height of the observation points is around 1000 m
we are efectivelly upward continuing the data.

.. jupyter-execute::

    grid_coords = vd.grid_coordinates(
        region=vd.get_region(coordinates),
        spacing=500,
        extra_coords=1500,
    )
    grid = eqs.grid(grid_coords, data_names=["magnetic_anomaly"])
    grid


.. jupyter-execute::

    fig = pygmt.Figure()

    title = "Observed magnetic anomaly data"
    pygmt.makecpt(
        cmap="polar+h0",
        series=(-maxabs, maxabs),
        background=True)

    with pygmt.config(FONT_TITLE="14p"):
        fig.plot(
            projection=fig_proj,
            region=xy_region,
            frame=[f"WSne+t{title}", "xa10000", "ya10000"],
            x=easting,
            y=northing,
            color=data.total_field_anomaly_nt,
            style="c0.1c",
            cmap=True,
        )
    fig.colorbar(cmap=True, frame=["a200f100", "x+lnT"])

    fig.shift_origin(xshift=fig_width + 1)

    title = "Gridded and upward-continued"

    with pygmt.config(FONT_TITLE="14p"):
        fig.grdimage(
            frame=[f"ESnw+t{title}", "xa10000", "ya10000"],
            grid=grid.magnetic_anomaly,
            cmap=True,
        )
    fig.colorbar(cmap=True, frame=["a200f100", "x+lnT"])

    fig.show()


----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <block-averaged-eqs>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <block-averaged-eqs>`
        :text-align: center
