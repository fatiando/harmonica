"""
Gravity corrections like Normal Gravity and Bouguer corrections.
"""
import numpy as np

from .ellipsoid import get_ellipsoid
from .constants import GRAVITATIONAL_CONST


def normal_gravity(latitude, height):  # pylint: disable=too-many-locals
    """
    Calculate normal gravity at any latitude and height.

    Computes the magnitude of the gradient of the gravity potential (gravitational +
    centrifugal) generated by a reference ellipsoid at the given latitude and
    (geometric) height. Uses of a closed form expression [LiGotze2001]_ and the current
    reference ellipsoid defined by :func:`harmonica.set_ellipsoid`.

    Parameters
    ----------
    latitude : float or array
        The (geodetic) latitude where the normal gravity will be computed (in degrees)
    height : float or array
        The ellipsoidal (geometric) height of computation point (in meters).

    Returns
    -------
    gamma : float or array
        The computed normal gravity (in mGal).

    """
    ell = get_ellipsoid()

    latitude_radians = np.deg2rad(latitude)
    sinlat = np.sin(latitude_radians)
    coslat = np.cos(latitude_radians)

    # The terms below follow the variable names in the Li and Goetze (2001) paper
    beta = np.arctan2(ell.semiminor_axis * sinlat / coslat, ell.semimajor_axis)
    zl2 = (ell.semiminor_axis * np.sin(beta) + height * sinlat) ** 2
    rl2 = (ell.semimajor_axis * np.cos(beta) + height * coslat) ** 2
    big_d = (rl2 - zl2) / ell.linear_eccentricity ** 2
    big_r = (rl2 + zl2) / ell.linear_eccentricity ** 2
    cosbeta_l2 = 0.5 * (1 + big_r) - np.sqrt(0.25 * (1 + big_r ** 2) - 0.5 * big_d)
    sinbeta_l2 = 1 - cosbeta_l2
    b_l = np.sqrt(rl2 + zl2 - ell.linear_eccentricity ** 2 * cosbeta_l2)
    q_0 = 0.5 * (
        (1 + 3 * (ell.semiminor_axis / ell.linear_eccentricity) ** 2)
        * np.arctan2(ell.linear_eccentricity, ell.semiminor_axis)
        - 3 * ell.semiminor_axis / ell.linear_eccentricity
    )
    q_l = (
        3
        * (1 + (b_l / ell.linear_eccentricity) ** 2)
        * (1 - b_l / ell.linear_eccentricity * np.arctan2(ell.linear_eccentricity, b_l))
        - 1
    )
    big_w = np.sqrt(
        (b_l ** 2 + ell.linear_eccentricity ** 2 * sinbeta_l2)
        / (b_l ** 2 + ell.linear_eccentricity ** 2)
    )

    # Put together gamma using 3 terms
    term1 = ell.geocentric_grav_const / (b_l ** 2 + ell.linear_eccentricity ** 2)
    term2 = (0.5 * sinbeta_l2 - 1 / 6) * (
        ell.semimajor_axis ** 2
        * ell.linear_eccentricity
        * q_l
        * ell.angular_velocity ** 2
        / ((b_l ** 2 + ell.linear_eccentricity ** 2) * q_0)
    )
    term3 = -cosbeta_l2 * b_l * ell.angular_velocity ** 2
    gamma = (term1 + term2 + term3) / big_w

    # Convert gamma from SI to mGal
    return gamma * 1e5


def bouguer_correction(topography, density_crust=2670, density_water=1040):
    r"""
    Gravitational effect of topography using a planar Bouguer plate approximation.

    Calculates the classic Bouguer correction term:

    .. math::

        g_{bg} = 2 \pi G \rho h

    in which :math:`G` is the gravitational constant and :math:`g_{bg}` is the
    gravitational effect of an infinite plate of thickness :math:`h` and density
    :math:`\rho`.

    Parameters
    ----------
    topography : array or :class:`xarray.DataArray`
        Topography height and bathymetry depth in meters. Should be referenced to the
        ellipsoid (ie, geometric heights).
    density_crust : float
        Density of the crust in :math:`kg/m^3`.
    density_water : float
        Water density in :math:`kg/m^3`.

    Returns
    -------
    grav_bouguer : array or :class:`xarray.DataArray`
         The gravitational effect of topography and residual bathymetry in mGal.

    """
    # Need to cast to array to make sure numpy indexing works as expected for 1D
    # DataArray topography
    oceans = np.array(topography < 0)
    continent = np.logical_not(oceans)
    density = np.full(topography.shape, np.nan, dtype="float")
    density[continent] = density_crust
    # The minus sign is used to negate the bathymetry (which is negative and the
    # equation calls for "thickness", not height). This is more practical than taking
    # the absolute value of the topography.
    density[oceans] = -1 * (density_water - density_crust)
    bouguer = 1e5 * 2 * np.pi * GRAVITATIONAL_CONST * density * topography
    return bouguer
