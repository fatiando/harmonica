# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Calculation of the IGRF magnetic field.
"""

import pathlib

import boule
import numba
import numpy as np
import pandas as pd
import pooch
import xarray as xr

from .._version import __version__
from . import legendre


def fetch_igrf14():
    """ """
    path = pooch.retrieve(
        url="doi:10.5281/zenodo.14218973/igrf14coeffs.txt",
        path=pooch.os_cache("harmonica"),
        known_hash="md5:3606931c15c9234d9ba8e2e91b729cb0",
    )
    return pathlib.Path(path)


def fetch_igrf13():
    """ """
    path = pooch.retrieve(
        url="doi:10.5281/zenodo.11269410/igrf13coeffs.txt",
        path=pooch.os_cache("harmonica"),
        known_hash="md5:e2e6e323086bde2dd910bb87d2db8532",
    )
    return pathlib.Path(path)


def load_igrf(path):
    """
    Load the IGRF Gauss coefficients from the classic text format.

    Parameters
    ----------
    path : str or :class:`pathlib.Path`
        The path to the ``.txt`` file with the Gauss coefficients.

    Returns
    -------
    years : array
        The years for the knot points of the time interpolation.

    """
    with open(path) as input_file:
        # Get rid of the comments and the first header line
        for line in input_file:
            if not line.startswith("#"):
                break
        # Read the years
        parts = input_file.readline().split()[3:-1]
        years = np.fromiter(parts + [float(parts[-1]) + 5], dtype="float")
        # Initialize the storage arrays
        max_degree = 13
        coeffs = {
            "g": np.zeros((years.size, max_degree + 1, max_degree + 1)),
            "h": np.zeros((years.size, max_degree + 1, max_degree + 1)),
        }
        # Read the coefficients
        for line in input_file:
            parts = line.split()
            key = parts[0].strip()
            degree, order = (int(i) for i in parts[1:3])
            coeffs[key][: years.size - 1, degree, order] = np.fromiter(
                parts[3:-1], dtype="float"
            )
            # Add the last year from the secular variation
            coeffs[key][years.size - 1, degree, order] = (
                float(parts[-1]) * 5 + coeffs[key][years.size - 2, degree, order]
            )
    return years, coeffs["g"], coeffs["h"]


def interpolate_coefficients(date, g, h, years):
    """
    Interpolate the coefficients to the given date.
    """
    g_date, h_date = None, None
    return g_date, h_date


class IGRF14:
    """
    14th generation of the International Geomagnetic Reference Field
    """

    def __init__(self, date, ellipsoid=boule.WGS84):
        self.date = date
        self._g, self._h = None, None
        self.reference_radius = 6371.2e3  # meters
        self.max_degree = 13

    @property
    def coefficients(self):
        "The Gauss coefficients g and h, respectively"
        if self._g is None or self._h is None:
            path = fetch_igrf13()
            g, h, years = load_igrf(path)
            self._g, self._h = interpolate_coefficients(date, g, h, years)
        return self._g, self._h

    @property
    def dipole_moment(self):
        """
        Dipole moment of the Earth on a geocentric Cartesian system
        """
        return mx, my, mz

    def predict(self, coordinates, field="b"):
        """
        Calculate the IGRF magnetic field at the given coordinates
        """
        longitude, latitude, height = coordinates
        longitude, latitude_sph, radius = self.ellipsoid.geodetic_to_spherical(
            longitude, latitude, height
        )
        longitude = np.radians(longitude)
        colatitude = np.radians(90 - latitude_sph)
        n_data = colatitude.size
        b_east = np.zeros(n_data)
        b_north = np.zeros(n_data)
        b_up = np.zeros(n_data)
        g, h = self.coefficients
        _evaluate_igrf(
            longitude, colatitude, radius, g, h, self.max_degree, b_east, b_north, b_up
        )
        return b_east, b_north, b_up


@numba.jit(parallel=True, nopython=True)
def _evaluate_igrf(
    longitude, colatitude, radius, g, h, max_degree, b_east, b_north, b_up
):
    n_data = longitude.size
    for i in numba.prange(n_data):
        plm = 1
