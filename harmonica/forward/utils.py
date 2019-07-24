"""
Utilities for forward modelling in Cartesian and spherical coordinates.
"""
import numpy as np
from numba import jit


def distance_cartesian(point_p, point_q):
    """
    Calculate the distance between two points given in Cartesian coordinates

    Parameters
    ----------
    point_p : list or tuple or 1d-array
        List, tuple or array containing the Cartesian coordinates of the first point
        in the following order: ``easting``, ``northing`` and ``down``.
        All quantities must be in meters.
    point_q : list or tuple or 1d-array
        List, tuple or array containing the Cartesian coordinates of the second point
        in the following order: ``easting``, ``northing`` and ``down``.
        All quantities must be in meters.

    Returns
    -------
    distance : float
        Distance between ``point_p`` and ``point_q``.
    """
    return np.sqrt(_distance_sq_cartesian(point_p, point_q))


@jit(nopython=True)
def _distance_sq_cartesian(point_p, point_q):
    """
    Calculate the square distance between two points given in Cartesian coordinates
    """
    easting, northing, down = point_p[:]
    easting_p, northing_p, down_p = point_q[:]
    distance_sq = (
        (easting - easting_p) ** 2 + (northing - northing_p) ** 2 + (down - down_p) ** 2
    )
    return distance_sq


def distance_spherical(point_p, point_q):
    """
    Calculate the distance between two points given in geocentric spherical coordinates

    Parameters
    ----------
    point_p : list or tuple or 1d-array
        List, tuple or array containing the coordinates of the first point in the
        following order: ``longitude``, ``latitude`` and ``radius``, given in a
        spherical geocentric coordiante system. Both ``longitude`` and
        ``latitude`` must be in degrees, while ``radius`` in meters.
    point_q : list or tuple or 1d-array
        List, tuple or array containing the coordinates of the second point in the
        following order: ``longitude``, ``latitude`` and ``radius``, given in a
        spherical geocentric coordiante system. Both ``longitude`` and
        ``latitude`` must be in degrees, while ``radius`` in meters.

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
    distance_sq, _, _ = _distance_sq_spherical(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    return np.sqrt(distance_sq)


@jit(nopython=True)
def _distance_sq_spherical(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Calculate the square distance between two points in spherical coordinates

    All angles must be in radians and radii in meters.

    Parameters
    ----------
    longitude, cosphi, sinphi, radius : floats
        Quantities related to the coordinates of the first point. ``cosphi`` and
        ``sinphi`` are the cosine and sine of the latitude coordinate of the first
        point, respectively. ``longitude`` must be in radians and ``radius`` in meters.
    longitude_p, cosphi_p, sinphi_p, radius_p : floats
        Quantities related to the coordinates of the second point. ``cosphi_p`` and
        ``sinphi_p`` are the cosine and sine of the latitude coordinate of the second
        point, respectively. ``longitude`` must be in radians and ``radius`` in meters.

    Returns
    -------
    distance_sq : float
        Square distance between the two points.
    cospsi : float
        Cosine of the psi angle.
    coslambda : float
        Cosine of the diference between the longitudes of both points.
    """
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance_sq = (radius - radius_p) ** 2 + 2 * radius * radius_p * (1 - cospsi)
    return distance_sq, cospsi, coslambda


# Jit compile distance_spherical and distance_cartesian for use in the numba functions
DISTANCE_CARTESIAN_NUMBA = jit(nopython=True)(distance_cartesian)
DISTANCE_SPHERICAL_NUMBA = jit(nopython=True)(distance_spherical)
