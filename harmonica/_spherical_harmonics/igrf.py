"""
Calculation of the IGRF magnetic field.
"""
import pathlib
import numpy as np
import numba
import xarray as xr
from . import legendre
from .._version import __version__

import pooch
import boule


def fetch_igrf13():
    """
    """
    path = pooch.retrieve(
        url="doi:10.5281/zenodo.11269410/igrf13coeffs.txt",
        path=pooch.os_cache("harmonica"),
        known_hash="md5:e2e6e323086bde2dd910bb87d2db8532",
    )
    return path


def load_igrf(path):
    """
    Load the IGRF Gauss coefficients from the given file
    """
    g, h, years = None, None, None
    return g, h, years


def interpolate_coefficients(date, g, h, years):
    """
    Interpolate the coefficients to the given date.
    """
    g_date, h_date = None, None
    return g_date, h_date


class IGRF13():
    """
    13th generation of the International Geomagnetic Reference Field
    """

    def __init__(self, date, ellipsoid=boule.WGS84):
        self.date = date
        path = fetch_igrf13()
        g, h, years = load_igrf(path)
        self._g, self._h = interpolate_coefficients(date, g, h, years)
        self.reference_radius = 6371.2e3  # meters

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
        latitude = coordinates[1]
        longitude, latitude_sph, radius = self.ellipsoid.geodetic_to_spherical(
            *coordinates
        )
        longitude = np.radians(longitude)
        colatitude = np.radians(90 - latitude_sph)


        return magnetic_field


@numba.jit(nopython=True)
def _cal
