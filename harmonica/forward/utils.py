"""
Utilities for forward modelling in Cartesian and spherical coordinates.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def distance_cartesian(point_a, point_b):
    """
    Calculate the distance between two points given in Cartesian coordinates

    Parameters
    ----------
    point_a : 1d-array
        Array containing the Cartesian coordinates of the first point in the following
        order: `easting`, `northing` and `vertical`.
        All quantities must be in meters.
    point_b : 1d-array
        Array containing the Cartesian coordinates of the second point in the following
        order: `easting`, `northing` and `vertical`.
        All quantities must be in meters.

    Returns
    -------
    distance : float
        Distance between ``point_a`` and ``point_b``.
    """
    return np.sqrt(_distance_sq_cartesian(point_a, point_b))


@jit(nopython=True)
def _distance_sq_cartesian(point_a, point_b):
    """
    Calculate the square distance between two points given in Cartesian coordinates
    """
    easting, northing, vertical = point_a[:]
    easting_p, northing_p, vertical_p = point_b[:]
    distance_sq = (
        (easting - easting_p) ** 2
        + (northing - northing_p) ** 2
        + (vertical - vertical_p) ** 2
    )
    return distance_sq


@jit(nopython=True)
def distance_spherical(point_a, point_b):
    """
    Calculate the distance between two points given in geocentric spherical coordinates

    Parameters
    ----------
    point_a : 1d-array
        Array containing the coordinates of the first point in the following order:
        `longitude`, `latitude` and `radius`, given in a spherical geocentric coordiante
        system.
        Both `longitude` and `latitude` must be in degrees, while `radius` in meters.
    point_b : 1d-array
        Array containing the coordinates of the second point in the following order:
        `longitude`, `latitude` and `radius`, given in a spherical geocentric coordiante
        system.
        Both `longitude` and `latitude` must be in degrees, while `radius` in meters.

    Returns
    -------
    distance : float
        Distance between ``point_a`` and ``point_b``.
    """
    # Get coordinates of the two points
    longitude, latitude, radius = point_a[:]
    longitude_p, latitude_p, radius_p = point_b[:]
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
    """
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance_sq = (radius - radius_p) ** 2 + 2 * radius * radius_p * (1 - cospsi)
    return distance_sq, cospsi, coslambda
