# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Calculation of the IGRF magnetic field.
"""
import datetime
import pathlib

import boule
import numba
import numpy as np
import pooch

from . import legendre


def fetch_igrf14():
    """ """
    path = pooch.retrieve(
        url="doi:10.5281/zenodo.14218973/igrf14coeffs.txt",
        path=pooch.os_cache("harmonica"),
        known_hash="md5:3606931c15c9234d9ba8e2e91b729cb0",
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
        years_str = input_file.readline().split()[3:-1]
        #   The years all have .0 at the end and int doesn't like it.
        years_int = [int(y.split(".")[0]) for y in years_str]
        #   Add another 5 years so we don't have to deal with interpolating
        #   differently when there is only the secular variation.
        years_int.append(years_int[-1] + 5)
        years = np.array(years_int, dtype="int")
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


def interpolate_coefficients(date, years, g, h):
    """
    Interpolate the coefficients to the given date.
    """
    index = int((date.year - years[0]) // 5)
    seconds_since_epoch = (
        date - datetime.datetime(year=years[index], month=1, day=1)
    ).total_seconds()
    epoch = (
        datetime.datetime(year=years[index] + 5, month=1, day=1)
        - datetime.datetime(year=years[index], month=1, day=1)
    ).total_seconds()
    g_date = np.zeros(g.shape[1:])
    h_date = np.zeros(h.shape[1:])
    max_n = g.shape[1]
    for n in range(max_n):
        for m in range(n):
            g_date[n, m] = (
                g[index, n, m]
                + seconds_since_epoch * (g[index + 1, n, m] - g[index, n, m]) / epoch
            )
            h_date[n, m] = (
                h[index, n, m]
                + seconds_since_epoch * (h[index + 1, n, m] - h[index, n, m]) / epoch
            )
    return g_date, h_date


class IGRF14:
    """
    International Geomagnetic Reference Field (14th generation).
    """

    def __init__(
        self, date, reference_radius=6371.2e3, max_degree=13, ellipsoid=boule.WGS84
    ):
        self.date = date
        self.reference_radius = reference_radius
        self.max_degree = max_degree
        self.ellipsoid = ellipsoid
        self._g, self._h = None, None

    def coefficients(self):
        """
        Load and interpolate the Gauss coefficients g and h.
        """
        if self._g is None or self._h is None:
            years, g, h = load_igrf(fetch_igrf14())
            self._g, self._h = interpolate_coefficients(self.date, years, g, h)
        return self._g, self._h

    def predict(self, coordinates, field="b"):
        """
        Calculate the IGRF magnetic field at the given coordinates
        """
        longitude, latitude, height = (np.atleast_1d(c) for c in coordinates)
        longitude, latitude_sph, radius = self.ellipsoid.geodetic_to_spherical(
            longitude, latitude, height
        )
        longitude_radians = np.radians(longitude)
        colatitude_radians = np.radians(90 - latitude_sph)

        n_data = longitude.size
        b_east = np.zeros(n_data)
        b_north_sph = np.zeros(n_data)
        b_radial = np.zeros(n_data)
        g, h = self.coefficients()
        _evaluate_igrf_spherical(
            longitude_radians,
            colatitude_radians,
            radius,
            g,
            h,
            self.max_degree,
            self.reference_radius,
            b_east,
            b_north_sph,
            b_radial,
        )
        # Rotate the vector from geocentric spherical to geodetic
        latitude_diff = -np.radians(latitude - latitude_sph)
        cos = np.cos(latitude_diff)
        sin = np.sin(latitude_diff)
        b_north = cos * b_north_sph + sin * b_radial
        b_up = -sin * b_north_sph + cos * b_radial
        return b_east, b_north, b_up


# @numba.jit(nopython=True)
def _evaluate_igrf_spherical(
    longitude,
    colatitude,
    radius,
    g,
    h,
    max_degree,
    reference_radius,
    b_east,
    b_north_sph,
    b_radial,
):
    n_data = longitude.size
    p = np.empty_like(g)
    p_deriv = np.empty_like(g)
    for i in range(n_data):
        legendre.associated_legendre_schmidt(np.cos(colatitude[i]), max_degree, p)
        legendre.associated_legendre_schmidt_derivative(max_degree, p, p_deriv)
        for n in range(1, max_degree + 1):
            r_frac = (reference_radius / radius[i]) ** (n + 2)
            for m in range(0, n + 1):
                cos = np.cos(m * longitude[i])
                sin = np.sin(m * longitude[i])
                b_east[i] += r_frac * (-m * g[n, m] * sin + m * h[n, m] * cos) * p[n, m]
                b_north_sph[i] += (
                    r_frac * (g[n, m] * cos + h[n, m] * sin) * p_deriv[n, m]
                )
                b_radial[i] += (
                    (n + 1) * r_frac * (g[n, m] * cos + h[n, m] * sin) * p[n, m]
                )
        b_east[i] *= -1 / np.sin(colatitude[i])
