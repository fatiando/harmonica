"""
Equivalent Layer interpolators for harmonic functions
"""
import numpy as np
from verde.base import BaseGridder, check_is_fitted, check_fit_input, least_squares


class HarmonicEQL(BaseGridder):
    """
    3D Equivalent Layer interpolator for harmonic fields using Green's functions
    """

    def __init__(self):
        """
        """

    def fit(self, coordinates, data, weights=None):
        """
        """
        coordinates, data, weights = check_fit_input(coordinates, data, weights)

    def predict(self, coordinates):
        """
        """
        # We know the gridder has been fitted if it has the mean
        check_is_fitted(self, ["mean_"])

    def jacobian(self, coordinates, point_masses):
        """
        """


def greens_func(coordinates, point_masses):
    """
    """
    east, north, vertical = coordinates[:3]
    east_p, north_p, vertical_p = point_masses[:]
    distance = np.sqrt(
        (east - east_p) ** 2
        + (north - north_p) ** 2
        + (vertical - vertical_p) ** 2
    )
    return 1 / distance
