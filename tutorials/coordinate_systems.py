"""
.. _coordinate_systems:

Coordinate Systems
==================

Gravity and magnetic data are usually observed at points above the Earth
surface, which can be referenced through different coordinate systems. When
processing these data, applying corrections, performing forward models and
inversions, we should take into account which type of coordiante system are we
using and if our choice introduces errors into these computations.

Harmonica can handle three different 3D coordinate systems:

- Cartesian coordinates
- Geodetic coordinates
- Spherical coordinates

"""

###############################################################################
# Cartesian coordinates
# ---------------------
#
# Cartesian (or plane) coordinates are referenced by a system defined by three
# orthogonal axis: two for the horizontal directions and one for the vertical
# one. Harmonica name the two horizontal axis as ``northing`` and ``easting``,
# while the vertical one is referenced as ``upward``, emphasizing the pointing
# direction. This means that positive values implies points above the zeroth
# plane, while negative values refer to points below. Harmonica assumes that
# the ``northing``, ``easting`` and ``upward`` coordinates are given in meters,
# unless it's specified.
#
# For example, we can define a prism below the zeroth horizontal plane and
# a regular grid of observation points at 100 meters above it

import verde as vd

# Define boundaries of the rectangular prism (in meters)
east, west, south, north = -20, 20, -20, 20
bottom, top = -40, -20
prism = [east, west, south, north, bottom, top]

# Define a regular grid of observation points (coordinates in meters)
coordinates = vd.grid_coordinates(
    region=(-40, 40, -40, 40), shape=(200, 200), extra_coords=100
)


###############################################################################
# Geodetic coordinates
# --------------------
#
# Geodetic (or geographic) coordinates use a reference ellipsoid for defining
# the ``longitude``, ``latitude`` and ``height`` coordinates of points. The
# reference ellipsoid is a mathematical surface that approximates the figure of
# the Earth and it's used to define point coordinates. Harmonica assumes that
# ``longitude``, ``latitude`` are given in degrees (except when noted), and the
# ellipsoidal height ``height`` is given in meters. Positive values of
# ``height`` refer to points outside the ellipsoid, while negative values refer
# to points that live inside it.
# Spatial data are usually given in geodetic coordinates, along with the
# reference ellipsoid on which they are defined.
#
# For example, let's define a regular grid of points (separated by equal
# angles) at 2km above the ellipsoid.

coordinates = vd.grid_coordiantes(
    region=(-70, -65, -35, -30), shape=(200, 200), extra_coords=2e3
)

# Some processes need to know the reference ellipsoid used to define the
# geodetic coordinates of points. Boule offers several ellipsoids that are
# commonly used on geophysical applications. Harmonica will ask for
# a ``boule.Ellipsoid`` instance every time it needs the reference ellipsoid.

import boule as bl

ellipsoid = bl.WGS84
print(ellipsoid)

# Some other processes are only designed to work under Cartesian coordinates.
# We can easily _convert_ geodetic coordinates to Cartesian by applying
# coordinates **projections**. This can be done through `pyproj`:

import pyproj

projection = pyproj.Proj(proj="merc", ellps="WGS84")
easting, northing = projection(*coordinates[:2])

projected_coordinates = (easting, northing, coordinates[-1])

# Remeber that this process implies projecting the geodetic coordinates onto
# a flat surface, what carries projection errors. For small regions, these
# errors are small, but for regional and global regions, these can heavily
# increase. Projections can also be used to recover geodetic coordiantes from
# Cartesian ones, by setting the ``inverse`` argument to ``True``.

###############################################################################
# Spherical coordiantes
# ---------------------
# Spherical coordinates are referenced by a coordinate system whose origin is
# located on the center of the Earth, that's why they are often called as
# spherical geocentric coordinates. From there, each point is specified through
# its ``longitude``, ``spherical_latitude`` and ``radius``, where the former
# ones are angles given in degrees and the latter is the distance between the
# point and the origin of the system (in meters).
# Although this reference system is rarely used for storing data, it's used for
# some non-Cartesian forward models, like Tesseroids (spherical prisms).
#
# Let's define a regular grid of points in spherical coordiantes, located at
# constant radius equal to the mean radius of the Earth.

coordinates = vd.grid_coordinates(
    region=(-70, -65, -35, -30), shape=(200, 200), extra_coords=ellipsoid.mean_radius
)

# We can convert spherical coordinates to geodetic ones through
# ``boule.Ellipsoid.spherical_to_geodetic``:

coordinates_geodetic = ellipsoid.spherical_to_geodetic(*coordinates)

# Remember that spherical and geodetic ``longitude`` are equivalent, while
# spherical and geodetic latitude will differ.
