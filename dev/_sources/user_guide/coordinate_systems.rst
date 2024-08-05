.. _coordinate_systems:

Coordinate systems
==================

Gravity and magnetic data are usually observed at points above the Earth
surface, which can be referenced through different coordinate systems.
When processing these data, applying corrections, performing forward models and
inversions, we should take into account which type of coordinate system are we
using and if our choice introduces errors into these computations.

Harmonica can handle three different 3D coordinate systems:

- :ref:`Cartesian coordinates <cartesian_coordinates>`
- :ref:`Geodetic coordinates <geodetic_coordinates>`
- :ref:`Spherical coordinates <spherical_coordinates>`

.. _cartesian_coordinates:

Cartesian coordinates
---------------------

Cartesian (or plain) coordinates are referenced by a system defined by three
orthogonal axis: two for the horizontal directions and one for the vertical
one. Harmonica name the two horizontal axis as *easting* and *northing*,
while the vertical one is referenced as *upward*, emphasizing its pointing
direction. This means that positive values implies points above the zeroth
plane, while negative values refer to points below. Harmonica assumes that
the *easting*, *northing* and *upward* coordinates are given in meters.

For example, we can define a prism below the horizontal plane at zeroth
height:

.. jupyter-execute::

    import verde as vd

    # Define boundaries of the rectangular prism (in meters)
    west, east, south, north = -20, 20, -20, 20
    bottom, top = -40, -20
    prism = [west, east, south, north, bottom, top]

Or a regular grid of points at 100 meters above the zeroth plane:

.. jupyter-execute::

    # Define a regular grid of observation points (coordinates in meters)
    coordinates = vd.grid_coordinates(
        region=(-40, 40, -40, 40), shape=(5, 5), extra_coords=100
    )
    easting, northing, upward = coordinates[:]

    print("easting:", easting)
    print("northing:", northing)
    print("upward:", upward)

Because the *upward* axis points in the upward direction, it's easy to check
if a given point is lower or higher than another one:

.. jupyter-execute::

    # Define two points
    point_1 = (30, 20, -67)
    point_2 = (30, 20, -58)
    print("Point 1 is higher than point 2?", point_1[2] > point_2[2])


.. _geodetic_coordinates:

Geodetic coordinates
--------------------

Geodetic (or geographic) coordinates use a reference ellipsoid for defining
the *longitude* (:math:`\lambda`), *latitude* (:math:`\varphi`) and *height*
(:math:`h`) coordinates of points (see :ref:`geodetic coordinates figure`).
The reference ellipsoid is a mathematical surface that approximates the
figure of the Earth and it's used to define point coordinates.

.. figure:: ../_static/figures/geodetic-coordinate-system.svg
   :name: geodetic coordinates figure
   :width: 90%
   :alt: Figure showing a reference ellipsoid along with an observation point "p". It also shows the geodetic coordinates of this observation point: its longitude, latitude and geodetic height.

   Figure: Geodetic coordinates

   Reference ellipsoid and a point **p** along with a geocentric Cartesian
   system (:math:`X`, :math:`Y`, :math:`Z`). Where :math:`a` and :math:`b`
   are the semimajor and semiminor axes of the ellipsoid, while the
   :math:`\lambda`, :math:`\varphi` and :math:`h` represent the geodetic
   coordinates of the point **p** in this geodetic coordinate system, where
   :math:`\lambda` is the *longitude*, :math:`\varphi` the *latitude* and
   :math:`h` the *height*.
   The :math:`\phi` is the *spherical latitude* of point **p** (see
   :ref:`spherical_coordinates`).
   This figure is a modified version of [Oliveira2021]_.


Harmonica assumes that *longitude*, *latitude* are given in decimal degrees
and the ellipsoidal height is given in meters. Positive values of *height*
refer to points outside the ellipsoid, while negative values refer to points
that live inside it.
Spatial data are usually given in geodetic coordinates, along with the
reference ellipsoid on which they are defined.

For example, let's define a regular grid of points (separated by equal
angles) at 2km above the ellipsoid using :mod:`verde`.

.. jupyter-execute::

    coordinates = vd.grid_coordinates(
        region=(-70, -65, -35, -30), shape=(6, 6), extra_coords=2e3
    )
    longitude, latitude, height = coordinates[:]
    print("longitude:", longitude)
    print("latitude:", latitude)
    print("height:", height)

Some processes need to know the reference ellipsoid used to define the
geodetic coordinates of points. :mod:`boule` offers several ellipsoids that
are commonly used on geophysical applications. Harmonica will ask for
a :class:`boule.Ellipsoid` instance as argument if it needs the reference
ellipsoid.

Lets define the WGS84 ellipsoide using :mod:`boule`:

.. jupyter-execute::

    import boule as bl

    ellipsoid = bl.WGS84
    print(ellipsoid)

Some other processes are only designed to work under Cartesian coordinates.
We can easily *transform* geodetic coordinates to Cartesian by applying
map **projections**. This can be done through :mod:`pyproj`.

As an example, lets project the *longitude* and *latitude* coordinates of the
previously generated grid using a Mercator projection:

.. jupyter-execute::

    import pyproj

    # Define a Mercator projection through pyproj
    projection = pyproj.Proj(proj="merc", ellps="WGS84")

    # Project the longitude and latitude coordinates of the grid points
    longitude, latitude = coordinates[:2]
    easting, northing = projection(longitude, latitude)

    print("easting:", easting)
    print("northing:", northing)

Remember that this process implies projecting the geodetic coordinates onto
a flat surface, what carries projection errors. For small regions, these
errors are small, but for regional and global regions, these can heavily
increase. Projections can also be used to recover geodetic coordinates from
Cartesian ones, by setting the ``inverse`` argument to ``True``.

.. _spherical_coordinates:

Spherical coordinates
---------------------

Spherical coordinates (a.k.a spherical geocentric coordinates) are defined
by a coordinate system whose origin is located on the center of the Earth.
Each point can be represented by its *longitude* (:math:`\lambda`),
*spherical latitude* (:math:`\phi`) and *radius* (:math:`r`) (see
:ref:`spherical coordinates figure`).

.. important::

   The *longitude* coordinates defined in *spherical coordinates* and in
   *geodetic coordinates* are equivalent.
   Nevertheless, the *spherical latitude* and the (geodetic) *latitude* are
   not. They would be the same if the reference ellipsoid were a sphere.

The *longitude* and *spherical latitude* are angles given in decimal degrees,
while the *radius* is the Euclidean distance between the point and the origin
of the system (in meters).
Although this reference system is rarely used for storing data, it's used for
some non-Cartesian forward models, like tesseroids (spherical prisms).

.. figure:: ../_static/figures/spherical-coordinate-system.svg
   :name: spherical coordinates figure
   :width: 50%
   :alt: Figure showing an observation point "p" defined in a spherical coordinate system.

   Figure: Spherical coordinates

   Point **p** defined in a spherical coordinate system, where
   :math:`\lambda` is the *longitude*, :math:`\phi` the *latitude* and
   :math:`r` the *radius*. The spherical coordinates are defined upon
   a geocentric Cartesian system (:math:`X`, :math:`Y`, :math:`Z`) whose
   origin is located in the center of the Earth.

Let's define a regular grid of points in spherical coordinates, located at
the same radius equal to the *mean radius of the Earth*.

.. jupyter-execute::

    coordinates = vd.grid_coordinates(
        region=(-70, -65, -35, -30),
        shape=(6, 6),
        extra_coords=ellipsoid.mean_radius,
    )
    longitude, sph_latitude, radius = coordinates[:]
    print("longitude:", longitude)
    print("spherical latitude:", sph_latitude)
    print("radius:", radius)

We can convert spherical coordinates to geodetic ones through
:meth:`boule.Ellipsoid.spherical_to_geodetic`:

.. jupyter-execute::

    coordinates_geodetic = ellipsoid.spherical_to_geodetic(*coordinates)
    longitude, latitude, height = coordinates_geodetic[:]
    print("longitude:", longitude)
    print("latitude:", latitude)
    print("height:", height)

While the conversion of spherical coordinates into geodetic ones can be
carried out through :meth:`boule.Ellipsoid.geodetic_to_spherical`:

.. jupyter-execute::

    coordinates_spherical = ellipsoid.geodetic_to_spherical(*coordinates_geodetic)
    longitude, sph_latitude, radius = coordinates_spherical[:]
    print("longitude:", longitude)
    print("spherical latitude:", sph_latitude)
    print("radius:", radius)

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <coordinate_systems>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <coordinate_systems>`
        :text-align: center
