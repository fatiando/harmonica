"""
Module for gravity corrections, normal gravity calculation, and processing routines.
"""

from .ellipsoid import get_ellipsoid


def normal_gravity(latitude, height):
    print(get_ellipsoid())
