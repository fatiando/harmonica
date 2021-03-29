# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utilities for forward modelling
"""
import numpy as np
from numba import jit


def distance(point_p, point_q, coordinate_system="cartesian", ellipsoid=None):
    """
    Distance between two points in Cartesian, spherical or geodetic coordinates

    Computes the Euclidean distance between two points given in Cartesian,
    spherical or geodetic coordinates.

    Parameters
    ----------
    point_p : list or tuple or 1d-array
        List, tuple or array containing the coordinates of the first point in
        the following order: (``easting``, ``northing`` and ``upward``) if
        given in Cartesian coordinates, (``longitude``, ``latitude`` and
        ``radius``) if given in a spherical geocentric coordiante system, or
        (``longitude``, ``latitude`` and ``height``) if given in geodetic
        coordinates.
        All ``easting``, ``northing`` and ``upward`` must be in meters.
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``radius``  and ``height`` in meters.
    point_q : list or tuple or 1d-array
        List, tuple or array containing the coordinates of the second point in
        the following order: (``easting``, ``northing`` and ``upward``) if
        given in Cartesian coordinates, (``longitude``, ``latitude`` and
        ``radius``) if given in a spherical geocentric coordiante system, or
        (``longitude``, ``latitude`` and ``height``) if given in geodetic
        coordinates.
        All ``easting``, ``northing`` and ``upward`` must be in meters.
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``radius``  and ``height`` in meters.
    coordinate_system : str (optional)
        Coordinate system of the coordinates of the computation points and the
        point masses.
        Available coordinates systems: ``cartesian``, ``spherical`` and
       ``geodetic``. Default ``cartesian``.
    ellipsoid : :class:`boule.Ellipsoid`
        Reference ellipsoid for points coordinates. Ignored if
        ``coordinate_system`` is not ``"geodetic"``. Default ``None``.

    Returns
    -------
    distance : float
        Distance between ``point_p`` and ``point_q``.
    """
    check_coordinate_system(coordinate_system)
    if coordinate_system == "cartesian":
        dist = distance_cartesian(point_p, point_q)
    if coordinate_system == "spherical":
        dist = distance_spherical(point_p, point_q)
    if coordinate_system == "geodetic":
        dist = distance_geodetic(point_p, point_q, ellipsoid)
    return dist


def check_coordinate_system(
    coordinate_system, valid_coord_systems=("cartesian", "spherical", "geodetic")
):
    """
    Check if the coordinate system is a valid one.

    Parameters
    ----------
    coordinate_system : str
        Coordinate system to be checked.
    valid_coord_system : tuple or list (optional)
        Tuple or list containing the valid coordinate systems.
        Default (``cartesian``, ``spherical``).
    """
    if coordinate_system not in valid_coord_systems:
        raise ValueError(
            "Coordinate system {} not recognized.".format(coordinate_system)
        )


@jit(nopython=True)
def distance_cartesian(point_p, point_q):
    """
    Calculate the distance between two points given in Cartesian coordinates

    Parameters
    ----------
    point_p : tuple or 1d-array
        Tuple or array containing the coordinates of the first point in the
        following order: (``easting``, ``northing`` and ``upward``)
        All coordinates must be in meters.
    point_q : tuple or 1d-array
        Tuple or array containing the coordinates of the second point in the
        following order: (``easting``, ``northing`` and ``upward``)
        All coordinates must be in meters.

    Returns
    -------
    distance : float
        Distance between ``point_p`` and ``point_q``.
    """
    easting, northing, upward = point_p[:]
    easting_p, northing_p, upward_p = point_q[:]
    dist = np.sqrt(
        (easting - easting_p) ** 2
        + (northing - northing_p) ** 2
        + (upward - upward_p) ** 2
    )
    return dist


@jit(nopython=True)
def distance_spherical(point_p, point_q):
    """
    Calculate the distance between two points in spherical coordinates

    All angles must be in degrees and radii in meters.

    Parameters
    ----------
    point_p : tuple or 1d-array
        Tuple or array containing the coordinates of the first point in the
        following order: (``longitude``, ``latitude`` and ``radius``).
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``radius`` in meters.
    point_q : tuple or 1d-array
        Tuple or array containing the coordinates of the second point in the
        following order: (``longitude``, ``latitude`` and ``radius``).
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``radius`` in meters.

    Returns
    -------
    distance : float
        Distance between ``point_p`` and ``point_q``.
    """
    # Get coordinates of the two points
    longitude, latitude, radius = point_p[:]
    longitude_p, latitude_p, radius_p = point_q[:]
    # Convert angles to radians
    longitude, latitude = np.radians(longitude), np.radians(latitude)
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    # Compute trigonometric quantities
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    dist, _, _ = distance_spherical_core(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    return dist


@jit(nopython=True)
def distance_spherical_core(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Core computation of distance between two points in spherical coordinates

    It computes the distance between two points in spherical coordinates given
    precomputed quantities related to the coordinates of both points: the
    ``longitude`` in radians, the sine and cosine of the ``latitude`` and the
    ``radius`` in meters. Precomputing this quantities may save computation
    time on some cases.

    Parameters
    ----------
    longitude, cosphi, sinphi, radius : floats
        Quantities related to the coordinates of the first point. ``cosphi``
        and ``sinphi`` are the cosine and sine of the latitude coordinate of
        the first point, respectively. ``longitude`` must be in radians and
        ``radius`` in meters.
    longitude_p, cosphi_p, sinphi_p, radius_p : floats
        Quantities related to the coordinates of the second point. ``cosphi_p``
        and ``sinphi_p`` are the cosine and sine of the latitude coordinate of
        the second point, respectively. ``longitude`` must be in radians and
        ``radius`` in meters.

    Returns
    -------
    distance : float
        Distance between the two points.
    cospsi : float
        Cosine of the psi angle.
    coslambda : float
        Cosine of the diference between the longitudes of both points.
    """
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    dist = np.sqrt((radius - radius_p) ** 2 + 2 * radius * radius_p * (1 - cospsi))
    return dist, cospsi, coslambda


def distance_geodetic(point_p, point_q, ellipsoid):  # pylint: disable=too-many-locals
    """
    Calculate the distance between two points in geodetic coordinates

    Computes the Euclidean distance between two points given in geodetic
    coordinates using the closed-form formula given by [Vajda2004]_.
    All angles must be in degrees and height above the ellipsoid in meters.

    Parameters
    ----------
    point_p : tuple or 1d-array
        Tuple or array containing the coordinates of the first point in the
        following order: (``longitude``, ``latitude`` and ``height``).
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``height`` in meters.
    point_q : tuple or 1d-array
        Tuple or array containing the coordinates of the second point in the
        following order: (``longitude``, ``latitude`` and ``height``).
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``height`` in meters.
    ellipsoid : :class:`boule.Ellipsoid`
        Reference ellipsoid for the geodetic coordinates of points ``point_p``
        and ``point_q``. Must be a instance of :class:`boule.Ellipsoid`.

    Returns
    -------
    distance : float
        Euclidean distance between ``point_p`` and ``point_q``.

    Example
    -------

    >>> import boule as bl
    >>> ellipsoid = bl.WGS84
    >>> point_p = (-72.3, -33.3, 644)
    >>> point_q = (-70.1, -31.6, 1024)
    >>> distance = distance_geodetic(point_p, point_q, ellipsoid)
    >>> print("{:.2f} m".format(distance))
    279878.84 m

    """
    # Get coordinates of the two points
    longitude, latitude, height = point_p[:]
    longitude_p, latitude_p, height_p = point_q[:]
    # Convert angles to radians
    longitude, latitude = np.radians(longitude), np.radians(latitude)
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    # Compute trigonometric quantities
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    coslambda = np.cos(longitude_p - longitude)
    # Compute prime vertical radii for both points
    prime_vertical_radius = ellipsoid.prime_vertical_radius(sinphi)
    prime_vertical_radius_p = ellipsoid.prime_vertical_radius(sinphi_p)
    # Compute the Euclidean distance using the close-form formula
    return geodetic_distance_core(
        cosphi,
        sinphi,
        height,
        cosphi_p,
        sinphi_p,
        height_p,
        coslambda,
        prime_vertical_radius,
        prime_vertical_radius_p,
        ellipsoid.first_eccentricity ** 2,
    )


def geodetic_distance_core(
    cosphi,
    sinphi,
    height,
    cosphi_p,
    sinphi_p,
    height_p,
    coslambda,
    prime_vertical_radius,
    prime_vertical_radius_p,
    ecc_sq,
):
    """
    Core computation of distance between two points in geodetic coordinates

    Parameters
    ----------
    cosphi, sinphi : floats
        Cosine and sine of the latitude angle for the first point
    height : float
        Height over ellipsoid of the first point (in meters).
    cosphi_p, sinphi_p : floats
        Cosine and sine of the latitude angle for the second point
    height_p : float
        Height over ellipsoid of the second point (in meters).
    coslambda : float
        Cosine of the difference between longitudes angles of both points.
    prime_vertical_radius : float
        Prime vertical radius for the latitude angle of the first point.
    prime_vertical_radius_p : float
        Prime vertical radius for the latitude angle of the second point.
    ecc_sq : float
        Square of ellipsoid first eccentricity.

    Returns
    -------
    distance : float
        Euclidean distance between both points.
    """
    upward_sum = prime_vertical_radius + height
    upward_sum_p = prime_vertical_radius_p + height_p
    dist = np.sqrt(
        upward_sum_p ** 2 * cosphi_p ** 2
        + upward_sum ** 2 * cosphi ** 2
        - 2 * upward_sum * upward_sum_p * cosphi * cosphi_p * coslambda
        + (upward_sum_p - ecc_sq * prime_vertical_radius_p) ** 2 * sinphi_p ** 2
        + (upward_sum - ecc_sq * prime_vertical_radius) ** 2 * sinphi ** 2
        - (
            2
            * (upward_sum_p - ecc_sq * prime_vertical_radius_p)
            * (upward_sum - ecc_sq * prime_vertical_radius)
            * sinphi
            * sinphi_p
        )
    )
    return dist
