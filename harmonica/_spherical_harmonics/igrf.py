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
        # The years all have .0 at the end and int doesn't like it.
        years = np.array([int(y.split(".")[0]) for y in years_str], dtype="int")
        # Initialize the storage arrays
        max_degree = 13
        coeffs = {
            "g": np.zeros((years.size, max_degree + 1, max_degree + 1)),
            "h": np.zeros((years.size, max_degree + 1, max_degree + 1)),
            "g_sv": np.zeros((max_degree + 1, max_degree + 1)),
            "h_sv": np.zeros((max_degree + 1, max_degree + 1)),
        }
        # Read the coefficients and the secular variation
        for line in input_file:
            parts = line.split()
            key = parts[0].strip()
            degree, order = (int(i) for i in parts[1:3])
            coeffs[key][: years.size, degree, order] = np.fromiter(
                parts[3:-1], dtype="float"
            )
            # Add secular variation to a different array
            coeffs[key + "_sv"][degree, order] = float(parts[-1])
    return years, coeffs


def interpolate_coefficients(date, years, coeffs):
    """
    Interpolate the coefficients to the given date.
    """
    if date.year < 1900:
        message = f"Invalid date {date} for IGRF. The model isn't valid before 1900."
        raise ValueError(message)
    # Find out which epoch is the last compatible one.
    index = int((date.year - years[0]) // 5)
    if index >= years.size:
        index = years.size - 1
    seconds_since_epoch = (
        date - datetime.datetime(year=years[index], month=1, day=1)
    ).total_seconds()
    epoch = (
        datetime.datetime(year=years[index] + 5, month=1, day=1)
        - datetime.datetime(year=years[index], month=1, day=1)
    ).total_seconds()
    year_in_seconds = 365.25 * 24 * 60 * 60
    g_date = np.zeros(coeffs["g"].shape[1:])
    h_date = np.zeros(coeffs["h"].shape[1:])
    for n in range(1, coeffs["g"].shape[1]):
        for m in range(n + 1):
            if index >= years.size - 1:
                variation_g = coeffs["g_sv"][n, m] / year_in_seconds
                variation_h = coeffs["h_sv"][n, m] / year_in_seconds
            else:
                variation_g = (
                    coeffs["g"][index + 1, n, m] - coeffs["g"][index, n, m]
                ) / epoch
                variation_h = (
                    coeffs["h"][index + 1, n, m] - coeffs["h"][index, n, m]
                ) / epoch
            g_date[n, m] = coeffs["g"][index, n, m] + seconds_since_epoch * variation_g
            h_date[n, m] = coeffs["h"][index, n, m] + seconds_since_epoch * variation_h
    return g_date, h_date


class IGRF14:
    """
    International Geomagnetic Reference Field (14th generation).
    """

    def __init__(
        self,
        date,
        reference_radius=6371.2e3,
        min_degree=1,
        max_degree=13,
        ellipsoid=boule.WGS84,
    ):
        if isinstance(date, str):
            self.date = datetime.datetime.fromisoformat(date)
        else:
            self.date = date
        self.reference_radius = reference_radius
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.ellipsoid = ellipsoid
        self._g, self._h = None, None

    @property
    def coefficients(self):
        """
        Load and interpolate the Gauss coefficients g and h.
        """
        if self._g is None or self._h is None:
            self._g, self._h = interpolate_coefficients(
                self.date, *load_igrf(fetch_igrf14())
            )
        return self._g, self._h

    def predict(self, coordinates):
        """
        Calculate the IGRF magnetic field at the given coordinates.
        """
        cast = np.broadcast(*coordinates[:3])
        longitude, latitude, height = (np.atleast_1d(c).ravel() for c in coordinates)
        longitude, latitude_sph, radius = self.ellipsoid.geodetic_to_spherical(
            longitude, latitude, height
        )
        longitude_radians = np.radians(longitude)
        colatitude_radians = np.radians(90 - latitude_sph)
        n_data = longitude.size
        b_east = np.zeros(n_data)
        b_north_sph = np.zeros(n_data)
        b_radial = np.zeros(n_data)
        g, h = self.coefficients
        _evaluate_igrf_spherical(
            longitude_radians,
            colatitude_radians,
            radius,
            g,
            h,
            self.min_degree,
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
        return (
            b_east.reshape(cast.shape),
            b_north.reshape(cast.shape),
            b_up.reshape(cast.shape),
        )


@numba.jit(nopython=True)
def _evaluate_igrf_spherical(
    longitude,
    colatitude,
    radius,
    g,
    h,
    min_degree,
    max_degree,
    reference_radius,
    b_east,
    b_north_sph,
    b_radial,
):
    """
    Calculate IGRF on discrete points using numba.
    """
    n_data = longitude.size
    p = np.zeros_like(g)
    p_deriv = np.zeros_like(g)
    for i in range(n_data):
        legendre.associated_legendre_schmidt(np.cos(colatitude[i]), max_degree, p)
        legendre.associated_legendre_schmidt_derivative(max_degree, p, p_deriv)
        for n in range(min_degree, max_degree + 1):
            r_frac = (reference_radius / radius[i]) ** (n + 2)
            for m in range(n + 1):
                # Try this to calculate the cos and sin:
                # https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Chebyshev_method
                cos = np.cos(m * longitude[i])
                sin = np.sin(m * longitude[i])
                b_east[i] += r_frac * (-m * g[n, m] * sin + m * h[n, m] * cos) * p[n, m]
                b_north_sph[i] += (
                    r_frac * (g[n, m] * cos + h[n, m] * sin) * p_deriv[n, m]
                )
                b_radial[i] += (
                    (n + 1) * r_frac * (g[n, m] * cos + h[n, m] * sin) * p[n, m]
                )
        sin_colat = np.sin(colatitude[i])
        # The east component is singular at the poles. Set it to zero if close
        # to the poles to avoid this.
        if abs(sin_colat) < 1e-10:
            b_east[i] = 0
        else:
            b_east[i] *= -1 / sin_colat
