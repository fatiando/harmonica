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
from .._version import __version__
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
        The years for the knot points (epochs) of the time interpolation.
    g, h : 3d-arrays
        The Gauss coefficients g and h as they vary through time. The first
        dimension is the epoch, the second is the degree n, and the third is
        the order m. For example, ``g[1, 3, 2]`` is the g coefficient at the
        second epoch for n=3 and m=2. At m > n, the coefficients are assigned
        zero. Units are nT.
    g_sv, h_sv : 2d-arrays
        The secular variation estimation for the 2 Gauss coefficients after the
        last epoch. The first dimension is the degree n and the second is the
        order m. At m > n, the coefficients are assigned zero. Units are
        nT/year.
    """
    with path.open() as input_file:
        # Get rid of the comments and the first header line
        for _ in range(3):
            input_file.readline()
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
    return years, coeffs["g"], coeffs["h"], coeffs["g_sv"], coeffs["h_sv"]


def interpolate_coefficients(date, years, g, h, g_sv, h_sv):
    """
    Interpolate the Gauss coefficients to the given date.

    Assumes that the time variation is piece wise linear. After the last year
    in the series, extrapolate using the estimated secular variation.

    Parameters
    ----------
    date : :class:`datetime.datetime`
        The date and time at which to interpolate the coefficients.
    years : 1d-array
        The years for the knot points (epochs) of the time interpolation.
    g, h : 3d-arrays
        The Gauss coefficients g and h as they vary through time. The first
        dimension is the epoch, the second is the degree n, and the third is
        the order m. For example, ``g[1, 3, 2]`` is the g coefficient at the
        second epoch for n=3 and m=2. At m > n, the coefficients are assigned
        zero. Units are nT.
    g_sv, h_sv : 2d-arrays
        The secular variation estimation for the 2 Gauss coefficients after the
        last epoch. The first dimension is the degree n and the second is the
        order m. At m > n, the coefficients are assigned zero. Units are
        nT/year.

    Returns
    -------
    g_date, h_date : 2d-arrays
        The interpolated Gauss coefficients. The first dimension is the degree
        n and the second is the order m. At m > n, the coefficients are
        assigned zero. Units are nT.
    """
    if date.year < years[0] or date.year >= years[-1] + 5:
        message = (
            f"Invalid date {date} for IGRF. "
            f"The model isn't valid before {years[0]} or on or after {years[-1] + 5}."
        )
        raise ValueError(message)
    # Find out which epoch is the last compatible one.
    index = int((date.year - years[0]) // 5)
    seconds_since_epoch = (
        date - datetime.datetime(year=years[index], month=1, day=1)
    ).total_seconds()
    epoch = (
        datetime.datetime(year=years[index] + 5, month=1, day=1)
        - datetime.datetime(year=years[index], month=1, day=1)
    ).total_seconds()
    year_in_seconds = 365.25 * 24 * 60 * 60
    g_date = np.zeros(g.shape[1:])
    h_date = np.zeros(h.shape[1:])
    for n in range(1, g.shape[1]):
        for m in range(n + 1):
            if index >= years.size - 1:
                variation_g = g_sv[n, m] / year_in_seconds
                variation_h = h_sv[n, m] / year_in_seconds
            else:
                variation_g = (g[index + 1, n, m] - g[index, n, m]) / epoch
                variation_h = (h[index + 1, n, m] - h[index, n, m]) / epoch
            g_date[n, m] = g[index, n, m] + seconds_since_epoch * variation_g
            h_date[n, m] = h[index, n, m] + seconds_since_epoch * variation_h
    return g_date, h_date


def calculate_ideal_spacing(max_degree):
    """
    Estimate the ideal spacing for a spherical harmonic synthesis.

    Uses the sampling theorem of [DriscollHealy1994]_ which provides a number
    of points to discretize an expansion with :math:`N` maximum degree as
    :math:`l = 2N + 2`.

    Parameters
    ----------
    max_degree : int
        The maximum degree of the expansion.

    Returns
    -------
    spacing : float
        The spacing in degrees.
    """
    latitude_range = 180
    return latitude_range / (2 * max_degree + 2)


class IGRF14:
    r"""
    International Geomagnetic Reference Field (14th generation).

    Calculate the three components of the magnetic field vector of the IGRF
    model in a geodetic (longitude, latitude, geometric height) system for any
    date between 1900 and 2030. Model coefficients are automatically downloaded
    and cached locally using :mod:`pooch`.

    .. note::

        The code is automatically parallelized (multithreaded shared-memory)
        with Numba. To evaluate the model on a regular grid, it's **at least 2x
        faster** to use the :meth:`~harmonica.IGRF14.grid` method than the
        :meth:`~harmonica.IGRF14.predict` method since the former is able to
        avoid repeat computations.

    Parameters
    ----------
    date : str or :class:`datetime.datetime`
        The date and time at which to calculate the IGRF field. If it's
        a string, should be an `ISO 8601 formatted date
        <https://en.wikipedia.org/wiki/ISO_8601>`__ and it will be converted
        into a Python :class:`datetime.datetime`. Must be on or after 1900 and
        before 2030.
    min_degree : int
        The minimum degree used in the expansion. Default is 1 (magnetic fields
        don't have the 0 degree term).
    max_degree : int
        The maximum degree used in the expansion. Default is 13.
    ellipsoid : :class:`boule.Ellipsoid`
        The ellipsoid used to convert geodetic to geocentric spherical
        coordinates and convert the magnetic field vector from a geocentric
        spherical to a geodetic system. Default is ``boule.WGS84``.

    Attributes
    ----------
    doi : str
        The DOI used to download the coefficient file.
    file_name : str
        The name of the coefficient file in the online archive.
    hash : str
        The hash of the coefficient file, used to check download integrity.
    reference_radius : float
        The reference radius used in the spherical harmonic expansion in
        meters.
    coefficients : tuple = (g, h)
        The g and h Gauss coefficients interpolated to the given date. Each
        coefficient is a 2d-array with shape ``(max_degree + 1, max_degree
        + 1)``. The degree n varies with rows and the order m varies with
        columns. The values where m > n are set to zero.

    References
    ----------
    [Alken2021]_

    [IAGA2024]_

    Notes
    -----
    The IGRF is a spherical harmonic model of the Earth's internal magnetic
    field. Its time variation is represented by piecewise linear functions. In
    practice, the model Gauss coefficients are provided in 5-year epochs and
    can be linearly interpolated between two epochs. For years later than the
    last epoch, the coefficients can be extrapolated linearly using the
    provided estimates of secular variation for each Gauss coefficient.

    The 3-component magnetic field vector in a geocentric spherical coordinate
    system (longitude, spherical colatitude, radius) system can be expressed in
    terms of spherical harmonics as:

    .. math::
        B_e(r, \theta, \lambda) = -\dfrac{1}{\sin\theta}
        \sum\limits_{n=1}^{N}\sum\limits_{m=0}^{n}
        \left(\dfrac{R}{r}\right)^{n+2} [ -m g_n^m \sin m\lambda + m h_n^m \cos
        m\lambda ] P_n^m(\cos\theta)

    .. math::
        B_n(r, \theta, \lambda) = \sum\limits_{n=1}^{N}\sum\limits_{m=0}^{n}
        \left(\dfrac{R}{r}\right)^{n+2} [ g_n^m \cos m\lambda + h_n^m \sin
        m\lambda ] \dfrac{\partial P_n^m(\cos\theta)}{\partial \theta}

    .. math::
        B_r(r, \theta, \lambda) = \sum\limits_{n=1}^{N}\sum\limits_{m=0}^{n} (n
        + 1)\left(\dfrac{R}{r}\right)^{n+2} [ g_n^m \cos m\lambda + h_n^m \sin
        m\lambda ] P_n^m(\cos\theta)

    in which :math:`B_e` is the easting/longitudinal component, :math:`B_n` is
    the northing/latitudinal component, :math:`B_r` is the radial component,
    :math:`r` is the radius coordinate, :math:`\theta` is the colatitude,
    :math:`\lambda` is the longitude, :math:`n` is the degree, :math:`m` is the
    order, :math:`P_n^m` are `associated Legendre functions
    <https://en.wikipedia.org/wiki/Associated_Legendre_polynomials>`__, and
    :math:`g_n^m` and  :math:`h_n^m` are the Gauss coefficients.

    The vector is converted to a geodetic system (longitude, latitude,
    height/upward) using the following rotation:

    .. math::
        B_{n}^{geodetic} = \cos(\varphi - \phi) B_n + \sin(\varphi - \phi) B_r

    .. math::
        B_{u} = -\sin(\varphi - \phi) B_n + \cos(\varphi - \phi) B_r

    in which :math:`\varphi` is the spherical latitude and :math:`\phi` is the
    geodetic latitude.
    """

    doi = "10.5281/zenodo.14218973"
    file_name = "igrf14coeffs.txt"
    hash = "md5:3606931c15c9234d9ba8e2e91b729cb0"
    reference_radius = 6371.2e3

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
        self._coefficients = None

    @property
    def coefficients(self):
        if self._coefficients is None:
            self._coefficients = interpolate_coefficients(
                self.date, *load_igrf(self._fetch_coefficient_file())
            )
        return self._coefficients

    def _fetch_coefficient_file(self):
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

    def predict(self, coordinates):
        """
        Calculate the IGRF magnetic field vector at the given coordinates.

        The field is evaluated using the given minimum and maximum degrees. The
        input coordinates are assumed to be in a geodetic coordinate system and
        are converted to geocentric spherical with the given ellipsoid. The
        output vector is rotated to the geodetic system with the horizontal
        components tangential to the ellipsoid and the upward component
        parallel to the ellipsoid normal.

        Parameters
        ----------
        coordinates : tuple = (longitude, latitude, height)
            Tuple of arrays with the longitude, latitude, and height
            coordinates of each point. Arrays can be Python lists or any
            numpy-compatible array type. Arrays can be of any shape but must
            all have the same shape. Longitude and latitude are in degrees and
            height in meters.

        Returns
        -------
        be, bn, bu : arrays
            The eastward, northward, and upward magnetic field vector
            components calculated at each point. The arrays will have the same
            shape as the coordinate arrays. All are in nT.
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
        b_vector = tuple(
            c.reshape(cast.shape)
            for c in vector_spherical_to_geodetic(
                latitude, latitude_sph, (b_east, b_north_sph, b_radial)
            )
        )
        return b_vector

    def grid(self, region, height, shape=None, spacing=None, adjust="spacing"):
        """
        Generate a grid of the IGRF magnetic field vector.

        Predict the magnetic field vector on a regular grid of geodetic
        coordinates. This is much faster than using
        :meth:`~harmonica.IGRF14.predict` to predict on the same points because
        we are able to reduce repeated calculations when we know the data are
        on a regular grid.

        If neither a spacing nor a shape are given, will estimate the optimal
        grid spacing for the maximum degree used using the sampling theorem of
        [DriscollHealy1994]_.

        Parameters
        ----------
        region : tuple = (W, E, S, N)
            The boundaries of the grid in geographic coordinates. Should have
            a lower and an upper boundary for each dimension of the coordinate
            system. If region is not given, will use the bounding region of the
            given coordinates. Units are degrees.
        height : 2d-array or float
            The geometric height above the reference ellipsoid where the
            magnetic field will be calculated. Units are meters.
        shape : tuple = (size_SN, size_WE) or None
            The number of points in each direction of the given region, in reverse
            order. Must have one integer value per dimension of the region. The
            order of arguments is the opposite of the order of the region for
            compatibility with numpy's ``.shape`` attribute. If None, *spacing*
            must be provided. Default is None.
        spacing : float, tuple = (space_SN, space_WE), or None
            The grid spacing in each direction of the given region, in reverse
            order. A single value means that the spacing is equal in all
            directions. If a tuple, must have one value per dimension of the
            region. The order of arguments is the opposite of the order of the
            region for compatibility with *shape*. If None, *shape* must be
            provided. Default is None. Units are degrees.
        adjust : str = "spacing" or "region"
            Whether to adjust the spacing or the region if the spacing is not
            a multiple of the region. Ignored if *shape* is given instead of
            *spacing*. Default is ``"spacing"``.

        Returns
        -------
        grid : :class:`xarray.Dataset`
            A collection of the grids of the eastward (``b_east``), northward
            (``b_north``), and upward (``b_up``) components of the magnetic
            field. Each has dimensions of latitude and longitude and the height
            as a non-dimensional coordinate. Grid points are grid-line
            registered.

        """
        if spacing is None and shape is None:
            spacing = calculate_ideal_spacing(self.max_degree)
        longitude, latitude, height = vd.grid_coordinates(
            region, spacing=spacing, shape=shape, adjust=adjust, extra_coords=height
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
        b_east, b_north, b_up = vector_spherical_to_geodetic(
            latitude, latitude_sph, (b_east, b_north_sph, b_radial)
        )
        grid = vd.make_xarray_grid(
            (longitude, latitude, height),
            (b_east, b_north, b_up),
            data_names=("b_east", "b_north", "b_up"),
            dims=("latitude", "longitude"),
            extra_coords_names="height",
        )
        grid.attrs["Conventions"] = "CF-1.8"
        grid.attrs["title"] = f"IGRF14 magnetic field at {self.date.isoformat()}"
        grid.attrs["crs"] = self.ellipsoid.name
        grid.attrs["source"] = (
            "Generated by spherical harmonic synthesis using library "
            f"Harmonica version {__version__}."
        )
        grid.attrs["description"] = (
            "Three components of the magnetic field vector "
            f"calculated at {self.date.isoformat()} "
            "in a geodetic coordinate system."
        )
        grid.attrs["units"] = "nT"
        grid.attrs["references"] = f"https://doi.org/{self.doi}"
        grid.longitude.attrs["long_name"] = "longitude"
        grid.longitude.attrs["standard_name"] = "longitude"
        grid.longitude.attrs["units"] = "degrees_east"
        grid.longitude.attrs["actual_range"] = (
            float(longitude.min()),
            float(longitude.max()),
        )
        grid.latitude.attrs["long_name"] = "latitude"
        grid.latitude.attrs["standard_name"] = "latitude"
        grid.latitude.attrs["units"] = "degrees_north"
        grid.latitude.attrs["actual_range"] = (
            float(latitude.min()),
            float(latitude.max()),
        )
        grid.height.attrs["long_name"] = "geometric height"
        grid.height.attrs["standard_name"] = "height_above_reference_ellipsoid"
        grid.height.attrs["units"] = "m"
        for component in ("b_east", "b_north", "b_up"):
            grid[component].attrs["units"] = "nT"
        grid.b_east.attrs["long_name"] = "Eastward component"
        grid.b_north.attrs["long_name"] = "Northward component"
        grid.b_up.attrs["long_name"] = "Upward component"
        grid.b_east.attrs["description"] = "Eastward component of the magnetic field"
        grid.b_north.attrs["description"] = "Northward component of the magnetic field"
        grid.b_up.attrs["description"] = "Upward component of the magnetic field"
        return grid


def vector_spherical_to_geodetic(latitude, latitude_spherical, vector):
    """
    Rotate a vector from a geocentric spherical to a geodetic system.

    Parameters
    ----------
    latitude : float or array
        The geodetic latitude of the vector in degrees.
    latitude_spherical : float or array
        The geocentric spherical latitude of the vector in degrees.
    vector : tuple = (east, north, radial)
        The 3 components of the vector along each of the spherical directions.

    Returns
    -------
    vector : tuple = (east, north, up)
        The 3 components of the rotated vector. The east component is
        unchanged.
    """
    rotation_angle = np.radians(latitude_spherical - latitude)
    cos = np.cos(rotation_angle)
    sin = np.sin(rotation_angle)
    east, north_sph, radial = vector
    north = cos * north_sph + sin * radial
    up = -sin * north_sph + cos * radial
    return (east, north, up)


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
