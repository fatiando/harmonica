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
import verde as vd

from .._utils import get_harmonica_cache
from . import legendre


def load_igrf(path):
    """
    Load the IGRF Gauss coefficients from the classic text format.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the ``.txt`` file with the Gauss coefficients.

    Returns
    -------
    years : 1d-array
        The years for the knot points of the time interpolation.
    coeffs : dict
        A dictionary with keys ``"g"`` and ``"h"`` which hold 3d-arrays that
        contain the Gauss coefficients and ``"g_sv"`` and ``"h_sv"`` which hold
        2d-arrays with the secular variation after the last year of the
        coefficient series.
    """
    with path.open() as input_file:
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
    Interpolate the Gauss coefficients to the given date.

    Assumes that the time variation is piece wise linear. After the last year
    in the series, extrapolate using the estimated secular variation.
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

    # The DOI used to download the coefficient file
    doi = "10.5281/zenodo.14218973"
    # The name of the file in the online archive
    file_name = "igrf14coeffs.txt"
    # The hash of the coefficient file
    hash = "md5:3606931c15c9234d9ba8e2e91b729cb0"
    # The reference radius used in the spherical harmonic expansion in meters
    reference_radius = (6371.2e3,)

    def __init__(
        self,
        date,
        min_degree=1,
        max_degree=13,
        ellipsoid=boule.WGS84,
    ):
        if isinstance(date, str):
            self.date = datetime.datetime.fromisoformat(date)
        else:
            self.date = date
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.ellipsoid = ellipsoid
        self._g, self._h = None, None

    def fetch_coefficient_file(self):
        """
        Download the coefficient file and cache it locally.

        Fetch it from an online source specified by the ``doi`` attribute of
        this class. If the file was already downloaded, it won't be downloaded
        again.

        Returns
        -------
        path : class:`pathlib.Path`
            Path to the downloaded file on disk.
        """
        path = pooch.retrieve(
            url=f"doi:{self.doi}/{self.file_name}",
            path=get_harmonica_cache(),
            known_hash=self.hash,
        )
        return pathlib.Path(path)

    @property
    def coefficients(self):
        """
        The g and h Gauss coefficients, interpolated to the given date.
        """
        if self._g is None or self._h is None:
            self._g, self._h = interpolate_coefficients(
                self.date, *load_igrf(self.fetch_coefficient_file())
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
        normalized_radius = self.reference_radius / radius
        n_data = longitude.size
        b_east = np.zeros(n_data)
        b_north_sph = np.zeros(n_data)
        b_radial = np.zeros(n_data)
        g, h = self.coefficients
        spherical_harmonics_magnetic_field(
            longitude_radians,
            colatitude_radians,
            normalized_radius,
            g,
            h,
            self.min_degree,
            self.max_degree,
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

    def grid(self, height, region, spacing=None, shape=None):
        """
        Calculate the IGRF magnetic field at the given coordinates.
        """
        longitude, latitude, height = vd.grid_coordinates(
            region, spacing=spacing, shape=shape, extra_coords=height
        )
        longitude, latitude_sph, radius = self.ellipsoid.geodetic_to_spherical(
            longitude, latitude, height
        )
        longitude_radians = np.radians(longitude)
        colatitude_radians = np.radians(90 - latitude_sph)
        normalized_radius = self.reference_radius / radius
        shape = longitude.shape
        b_east = np.zeros(shape)
        b_north_sph = np.zeros(shape)
        b_radial = np.zeros(shape)
        g, h = self.coefficients
        spherical_harmonics_magnetic_field_grid(
            longitude_radians[0, :],
            colatitude_radians[:, 0],
            normalized_radius,
            g,
            h,
            self.min_degree,
            self.max_degree,
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
        grid = vd.make_xarray_grid(
            (longitude, latitude, height),
            (b_east, b_north, b_up),
            data_names=("b_east", "b_north", "b_up"),
            dims=("latitude", "longitude"),
            extra_coords_names="height",
        )
        return grid


@numba.jit(nopython=True, parallel=True)
def spherical_harmonics_magnetic_field(
    longitude,
    colatitude,
    normalized_radius,
    g,
    h,
    min_degree,
    max_degree,
    b_east,
    b_north_sph,
    b_radial,
):
    """
    Calculate a spherical harmonic expansion of a magnetic field.
    """
    n_data = longitude.size
    for i in numba.prange(n_data):
        cos_colat = np.cos(colatitude[i])
        sin_colat = np.sin(colatitude[i])
        # Have to allocate here because of the parallel loop. These are small
        # for low degree so not a huge time sink.
        p = np.empty_like(g)
        p_deriv = np.empty_like(g)
        legendre.associated_legendre_schmidt(cos_colat, max_degree, p)
        legendre.associated_legendre_schmidt_derivative(max_degree, p, p_deriv)
        # Pre-compute the sin and cos of longitude to avoid repeated
        # computation for every value of n. Use the recursive Chebyshev method
        # to calculate sin/cos(m lon) to save running trig functions. This
        # method is about 20% faster than running the sin and cos several
        # times. See:
        # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
        cos_mlon = np.empty(max_degree + 1)
        sin_mlon = np.empty(max_degree + 1)
        cos_mlon[0] = 1
        sin_mlon[0] = 0
        cos_mlon[1] = np.cos(longitude[i])
        sin_mlon[1] = np.sin(longitude[i])
        for m in range(2, max_degree + 1):
            cos_mlon[m] = 2 * cos_mlon[1] * cos_mlon[m - 1] - cos_mlon[m - 2]
            sin_mlon[m] = 2 * cos_mlon[1] * sin_mlon[m - 1] - sin_mlon[m - 2]
        # Calculating the power like this results in about an 8% performance
        # boost. Not much but I'll take it.
        r_frac = normalized_radius[i] ** (min_degree - 1 + 2)
        for n in range(min_degree, max_degree + 1):
            r_frac *= normalized_radius[i]
            for m in range(n + 1):
                b_east[i] += (
                    r_frac
                    * (-m * g[n, m] * sin_mlon[m] + m * h[n, m] * cos_mlon[m])
                    * p[n, m]
                )
                b_north_sph[i] += (
                    r_frac
                    * (g[n, m] * cos_mlon[m] + h[n, m] * sin_mlon[m])
                    * p_deriv[n, m]
                )
                b_radial[i] += (
                    (n + 1)
                    * r_frac
                    * (g[n, m] * cos_mlon[m] + h[n, m] * sin_mlon[m])
                    * p[n, m]
                )
        # The east component is singular at the poles. Set it to zero if close
        # to the poles to avoid this.
        if abs(sin_colat) < 1e-10:
            b_east[i] = 0
        else:
            b_east[i] *= -1 / sin_colat


@numba.jit(nopython=True, parallel=True)
def spherical_harmonics_magnetic_field_grid(
    longitude,
    colatitude,
    normalized_radius,
    g,
    h,
    min_degree,
    max_degree,
    b_east,
    b_north_sph,
    b_radial,
):
    """
    Calculate a spherical harmonic expansion of a magnetic field.
    """
    n_lon = longitude.size
    n_colat = colatitude.size
    # Pre-compute the sin and cos of longitude to avoid repeated computation.
    # Use the recursive Chebyshev method to calculate sin/cos(m lon) to save
    # running trig functions. See:
    # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
    cos_mlon = np.empty((n_lon, max_degree + 1))
    sin_mlon = np.empty((n_lon, max_degree + 1))
    for j in numba.prange(n_lon):
        cos_mlon[j, 0] = 1
        sin_mlon[j, 0] = 0
        cos_mlon[j, 1] = np.cos(longitude[j])
        sin_mlon[j, 1] = np.sin(longitude[j])
        for m in range(2, max_degree + 1):
            cos_mlon[j, m] = (
                2 * cos_mlon[j, 1] * cos_mlon[j, m - 1] - cos_mlon[j, m - 2]
            )
            sin_mlon[j, m] = (
                2 * cos_mlon[j, 1] * sin_mlon[j, m - 1] - sin_mlon[j, m - 2]
            )
    for i in numba.prange(n_colat):
        cos_colat = np.cos(colatitude[i])
        sin_colat = np.sin(colatitude[i])
        # Have to allocate here because of the parallel loop. These are small
        # for low degree so not a huge time sink.
        p = np.empty_like(g)
        p_deriv = np.empty_like(g)
        legendre.associated_legendre_schmidt(cos_colat, max_degree, p)
        legendre.associated_legendre_schmidt_derivative(max_degree, p, p_deriv)
        for j in range(n_lon):
            r_frac = normalized_radius[i, j] ** (min_degree - 1 + 2)
            for n in range(min_degree, max_degree + 1):
                r_frac *= normalized_radius[i, j]
                for m in range(n + 1):
                    b_east[i, j] += (
                        r_frac
                        * (-m * g[n, m] * sin_mlon[j, m] + m * h[n, m] * cos_mlon[j, m])
                        * p[n, m]
                    )
                    b_north_sph[i, j] += (
                        r_frac
                        * (g[n, m] * cos_mlon[j, m] + h[n, m] * sin_mlon[j, m])
                        * p_deriv[n, m]
                    )
                    b_radial[i, j] += (
                        (n + 1)
                        * r_frac
                        * (g[n, m] * cos_mlon[j, m] + h[n, m] * sin_mlon[j, m])
                        * p[n, m]
                    )
            # The east component is singular at the poles. Set it to zero if close
            # to the poles to avoid this.
            if abs(sin_colat) < 1e-10:
                b_east[i, j] = 0
            else:
                b_east[i, j] *= -1 / sin_colat
