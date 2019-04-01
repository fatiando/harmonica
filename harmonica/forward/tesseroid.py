"""
Forward modelling for tesseroids
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST

STACK_SIZE = 100


@jit(nopython=True)
def adaptive_discretization(
    coordinates,
    tesseroid,
    distance_size_ratio,
    stack_size=STACK_SIZE,
    radial_discretization=False,
):
    """
    Three or two dimensional adaptive discretization
    """
    # Create list of small tesseroids
    small_tesseroids = []
    # Create stack of tesseroids
    stack = np.zeros((stack_size, 6))
    stack[0, :] = tesseroid
    stack_top = 0
    while stack_top >= 0:
        # Pop the first tesseroid from the stack
        tesseroid = [stack[stack_top, i] for i in range(6)]
        stack_top -= 1
        # Get its dimensions
        l_lon, l_lat, l_rad = _tesseroid_dimensions(tesseroid)
        # Get distance between computation point and center of tesseroid
        distance = _distance_tesseroid_point(coordinates, tesseroid)
        # Check inequality
        n_lon, n_lat, n_rad = 1, 1, 1
        if distance / l_lon < distance_size_ratio:
            n_lon = 2
        if distance / l_lat < distance_size_ratio:
            n_lat = 2
        if distance / l_rad < distance_size_ratio and radial_discretization:
            n_rad = 2
        # Apply discretization
        if n_lon * n_lat * n_rad > 1:
            stack_top = _split_tesseroid(
                tesseroid, n_lon, n_lat, n_rad, stack, stack_top
            )
        else:
            small_tesseroids.append(tesseroid)
    return np.array(small_tesseroids)


@jit(nopython=True)
def _split_tesseroid(tesseroid, n_lon, n_lat, n_rad, stack, stack_top):
    """
    Split tesseroid along each dimension
    """
    w, e, s, n, bottom, top = tesseroid[:]
    if stack_top + n_lon * n_lat * n_rad > stack.shape[0]:
        raise OverflowError("Tesseroid stack overflow.")
    # Compute differential distance
    # These lines may give errors while working near the 0 - 360 boundary
    d_lon = (e - w) / n_lon
    d_lat = (n - s) / n_lat
    d_rad = (top - bottom) / n_rad
    for i in range(n_lon):
        for j in range(n_lat):
            for k in range(n_rad):
                stack_top += 1
                stack[stack_top, 0] = w + d_lon * i
                stack[stack_top, 1] = w + d_lon * (i + 1)
                stack[stack_top, 2] = s + d_lat * j
                stack[stack_top, 3] = s + d_lat * (j + 1)
                stack[stack_top, 4] = bottom + d_rad * k
                stack[stack_top, 5] = bottom + d_rad * (k + 1)
    return stack_top


@jit(nopython=True)
def _tesseroid_dimensions(tesseroid):
    """
    Calculate the dimensions of the tesseroid.
    """
    w, e, s, n, bottom, top = tesseroid[:]
    w, e, s, n = np.radians(w), np.radians(e), np.radians(s), np.radians(n)
    latitude_center = (n + s) / 2
    l_lat = top * np.arccos(np.sin(n) * np.sin(s) + np.cos(n) * np.cos(s))
    l_lon = top * np.arccos(
        np.sin(latitude_center) ** 2 + np.cos(latitude_center) ** 2 * np.cos(e - w)
    )
    l_rad = top - bottom
    return l_lon, l_lat, l_rad


@jit(nopython=True)
def _distance_tesseroid_point(coordinates, tesseroid):
    """
    Calculate the distance between a computation point and the center of a tesseroid.
    """
    longitude, latitude, radius = coordinates[:]
    longitude, latitude = np.radians(longitude), np.radians(latitude)
    # Get center of the tesseroid
    longitude_p = (tesseroid[0] + tesseroid[1]) / 2
    latitude_p = (tesseroid[2] + tesseroid[3]) / 2
    radius_p = (tesseroid[4] + tesseroid[5]) / 2
    # Convert angles to radians
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    coslambda = np.cos(longitude_p - longitude)
    cospsi = sinphi_p * sinphi + cosphi_p * cosphi * coslambda
    distance = np.sqrt(radius ** 2 + radius_p ** 2 - 2 * radius * radius_p * cospsi)
    return distance
