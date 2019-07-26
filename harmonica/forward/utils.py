"""
Utilities for forward modelling in Cartesian and spherical coordinates.
"""
import numpy as np
from numba import jit


def distance(point_p, point_q, coordinate_system="cartesian"):
    """
    Calculate the distance between two points in Cartesian or spherical coordinates

    Parameters
    ----------
    point_p : list or tuple or 1d-array
        List, tuple or array containing the coordinates of the first point in the
        following order: (``easting``, ``northing`` and ``down``) if given in Cartesian
        coordinates, or (``longitude``, ``latitude`` and ``radius``) if given in
        a spherical geocentric coordiante system.
        All ``easting``, ``northing`` and ``down`` must be in meters.
        Both ``longitude`` and ``latitude`` must be in degrees, while ``radius`` in
        meters.
    point_q : list or tuple or 1d-array
        List, tuple or array containing the coordinates of the second point in the
        following order: (``easting``, ``northing`` and ``down``) if given in Cartesian
        coordinates, or (``longitude``, ``latitude`` and ``radius``) if given in
        a spherical geocentric coordiante system.
        All ``easting``, ``northing`` and ``down`` must be in meters.
        Both ``longitude`` and ``latitude`` must be in degrees, while ``radius`` in
        meters.
    coordinate_system : str (optional)
        Coordinate system of the coordinates of the computation points and the point
        masses. Available coordinates systems: ``cartesian``, ``spherical``.
        Default ``cartesian``.

    Returns
    -------
    distance : float
        Distance between ``point_p`` and ``point_q``.
    """
    if coordinate_system not in ("cartesian", "spherical"):
        raise ValueError(
            "Coordinate system {} not recognized.".format(coordinate_system)
        )
    if coordinate_system == "cartesian":
        dist = np.sqrt(_distance_sq_cartesian(point_p, point_q))
    elif coordinate_system == "spherical":
        dist = np.sqrt(_distance_sq_spherical(point_p, point_q))
    return dist


@jit(nopython=True)
def _distance_sq_cartesian(point_p, point_q):
    """
    Calculate the square distance between two points given in Cartesian coordinates

    Parameters
    ----------
    point_p : tuple or 1d-array
        Tuple or array containing the coordinates of the first point in the
        following order: (``easting``, ``northing`` and ``down``)
        All coordinates must be in meters.
    point_q : tuple or 1d-array
        Tuple or array containing the coordinates of the second point in the
        following order: (``easting``, ``northing`` and ``down``)
        All coordinates must be in meters.

    Returns
    -------
    distance_sq : float
        Square distance between ``point_p`` and ``point_q``.
    """
    easting, northing, down = point_p[:]
    easting_p, northing_p, down_p = point_q[:]
    distance_sq = (
        (easting - easting_p) ** 2 + (northing - northing_p) ** 2 + (down - down_p) ** 2
    )
    return distance_sq


@jit(nopython=True)
def _distance_sq_spherical(point_p, point_q):
    """
    Calculate the square distance between two points in spherical coordinates

    All angles must be in degrees and radii in meters.

    Parameters
    ----------
    point_p : tuple or 1d-array
        Tuple or array containing the coordinates of the first point in the
        following order: (``longitude``, ``latitude`` and ``radius``).
        Both ``longitude`` and ``latitude`` must be in degrees, while ``radius`` in
        meters.
    point_q : tuple or 1d-array
        Tuple or array containing the coordinates of the second point in the
        following order: (``longitude``, ``latitude`` and ``radius``).
        Both ``longitude`` and ``latitude`` must be in degrees, while ``radius`` in
        meters.

    Returns
    -------
    distance_sq : float
        Square distance between ``point_p`` and ``point_q``.
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
    distance_sq, _, _ = _distance_sq_spherical_core(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    return distance_sq


@jit(nopython=True)
def _distance_sq_spherical_core(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Core computation for the square distance between two points in spherical coordinates

    It computes the square distance between two points in spherical coordinates given
    precomputed quantities related to the coordinates of both points: the ``longitude``
    in radians, the sine and cosine of the ``latitude`` and the ``radius`` in meters.
    Precomputing this quantities may save computation time on some cases.

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
