"""
Forward modelling for tesseroids
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST


def adaptive_discretization(coordinates, tesseroid, distance_size_ratio):
    """
    Two dimensional adaptive discretization
    """
    # Create list of small tesseroids
    small_tesseroids = []
    # Create stack of tesseroids
    stack = [tesseroid]
    while stack:
        # Pop the first tesseroid from the stack
        tesseroid = stack.pop()
        # Get its dimensions
        L_lon, L_lat = _tesseroid_horizontal_dimensions(tesseroid)
        # Get distance between computation point and center of tesseroid
        distance = _distance_tesseroid_point(coordinates, tesseroid)
        # Check inequality
        split_lon = bool(distance / L_lon < distance_size_ratio)
        split_lat = bool(distance / L_lat < distance_size_ratio)
        if split_lon or split_lat:
            _split_tesseroid(tesseroid, split_lon, split_lat, stack)
        else:
            small_tesseroids.append(tesseroid)
    return np.array(small_tesseroids)


def _split_tesseroid(tesseroid, split_lon, split_lat, stack):
    """
    Split tesseroid along horizontal dimensions
    """
    w, e, s, n, bottom, top = tesseroid[:]
    n_lon, n_lat = 1, 1
    if split_lon:
        n_lon = 2
    if split_lat:
        n_lat = 2
    d_lon = (e - w) / n_lon
    d_lat = (n - s) / n_lat
    for i in range(n_lon):
        for j in range(n_lat):
            stack.append(
                [
                    w + d_lon * i,
                    w + d_lon * (i + 1),
                    s + d_lat * j,
                    s + d_lat * (j + 1),
                    bottom,
                    top,
                ]
            )


def _tesseroid_horizontal_dimensions(tesseroid):
    """
    Calculate the horizontal dimensions of the tesseroid.
    """
    w, e, s, n = np.radians(np.array(tesseroid[:4]))
    bottom, top = tesseroid[-2:]
    latitude_center = (n + s) / 2
    L_lat = top * np.arccos(np.sin(n) * np.sin(s) + np.cos(n) * np.cos(s))
    L_lon = top * np.arccos(
        np.sin(latitude_center) ** 2 + np.cos(latitude_center) ** 2 * np.cos(e - w)
    )
    return L_lon, L_lat


def _distance_tesseroid_point(coordinates, tesseroid):
    """
    Calculate the distance between a computation point and the center of a tesseroid.
    """
    longitude, latitude, radius = coordinates[:]
    w, e, s, n, bottom, top = tesseroid[:]
    # Get center of the tesseroid
    longitude_p = (e + w) / 2
    latitude_p = (n + s) / 2
    radius_p = (top + bottom) / 2
    # Convert angles to radians
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    radius_p_sq = radius_p ** 2
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance = np.sqrt(radius ** 2 + radius_p_sq - 2 * radius * radius_p * cospsi)
    return distance
