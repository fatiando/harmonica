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
        L_lon, L_lat, L_r = _tesseroid_dimensions(tesseroid)
        # Get distance between computation point and center of tesseroid
        distance = _distance_tesseroid_point(coordinates, tesseroid)
        # Check inequality
        split_lon = bool(distance / L_lon < distance_size_ratio)
        split_lat = bool(distance / L_lat < distance_size_ratio)
        split_radial = bool(distance / L_r < distance_size_ratio)
        # Choose 2D or 3D adaptive discretization
        if not radial_discretization:
            split_radial = False
        # Apply discretization
        if split_lon or split_lat or split_radial:
            stack_top = _split_tesseroid(
                tesseroid, split_lon, split_lat, split_radial, stack, stack_top
            )
        else:
            small_tesseroids.append(tesseroid)
    return np.array(small_tesseroids)


@jit(nopython=True)
def _split_tesseroid(
    tesseroid,
    split_lon,
    split_lat,
    split_radial,
    stack,
    stack_top,
    stack_size,
):
    """
    Split tesseroid along each dimension
    """
    w, e, s, n, bottom, top = tesseroid[:]
    n_lon, n_lat = 1, 1
    if split_lon:
        n_lon = 2
    if split_lat:
        n_lat = 2
    if split_radial:
        n_radial = 2
    if stack_top + n_lon * n_lat * n_radial > stack_size:
        raise OverflowError("Tesseroid stack overflow.")
    # Compute differential distance
    # These lines may give errors while working near the 0 - 360 boundary
    d_lon = (e - w) / n_lon
    d_lat = (n - s) / n_lat
    d_radial = (top - bottom) / n_radial
    for i in range(n_lon):
        for j in range(n_lat):
            for k in range(n_radial):
                stack_top += 1
                stack[stack_top, 0] = w + d_lon * i
                stack[stack_top, 1] = w + d_lon * (i + 1)
                stack[stack_top, 2] = s + d_lat * j
                stack[stack_top, 3] = s + d_lat * (j + 1)
                stack[stack_top, 4] = bottom + d_radial * k
                stack[stack_top, 5] = bottom + d_radial * (k + 1)
    return stack_top


@jit(nopython=True)
def _tesseroid_dimensions(tesseroid):
    """
    Calculate the dimensions of the tesseroid.
    """
    w, e, s, n, bottom, top = tesseroid[:]
    w, e, s, n = np.radians(w), np.radians(e), np.radians(s), np.radians(n)
    latitude_center = (n + s) / 2
    L_lat = top * np.arccos(np.sin(n) * np.sin(s) + np.cos(n) * np.cos(s))
    L_lon = top * np.arccos(
        np.sin(latitude_center) ** 2 + np.cos(latitude_center) ** 2 * np.cos(e - w)
    )
    L_r = top - bottom
    return L_lon, L_lat, L_r


@jit(nopython=True)
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
