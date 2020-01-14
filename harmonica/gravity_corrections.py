"""
Gravity corrections like Normal Gravity and Bouguer corrections.
"""
import numpy as np

from .constants import GRAVITATIONAL_CONST


def bouguer_correction(topography, density_crust=2670, density_water=1040):
    r"""
    Gravitational effect of topography using a Bouguer plate approximation

    Used to remove the gravitational attraction of topography above the
    ellipsoid from the gravity disturbance. The infinite plate approximation is
    adequate for regions with flat topography and observation points close to
    the surface of the Earth.

    This function calculates the classic Bouguer correction:

    .. math::

        g_{bg} = 2 \pi G \rho h

    in which :math:`G` is the gravitational constant and :math:`g_{bg}` is the
    gravitational effect of an infinite plate of thickness :math:`h` and
    density :math:`\rho`.

    In the oceans, subtracting normal gravity from the observed gravity results
    in over correction because the normal Earth has crust where there was water
    in the real Earth. The Bouguer correction for the oceans aims to remove
    this residual effect due to the over correction:

    .. math::

        g_{bg} = 2 \pi G (\rho_w - \rho_c) |h|

    in which :math:`\rho_w` is the density of water and :math:`\rho_c` is the
    density of the crust of the normal Earth. We need to take the absolute
    value of the bathymetry :math:`h` because it is negative and the equation
    requires a thickness value (positive).

    Parameters
    ----------
    topography : array or :class:`xarray.DataArray`
        Topography height and bathymetry depth in meters.
        Should be referenced to the ellipsoid (ie, geometric heights).
    density_crust : float
        Density of the crust in :math:`kg/m^3`.
        Used as the density of topography on land and the density of the normal
        Earth's crust in the oceans.
    density_water : float
        Density of water in :math:`kg/m^3`.

    Returns
    -------
    grav_bouguer : array or :class:`xarray.DataArray`
        The gravitational effect of topography and residual bathymetry in mGal.

    """
    # Need to cast to array to make sure numpy indexing works as expected for
    # 1D DataArray topography
    oceans = np.array(topography < 0)
    continent = np.logical_not(oceans)
    density = np.full(topography.shape, np.nan, dtype="float")
    density[continent] = density_crust
    # The minus sign is used to negate the bathymetry (which is negative and
    # the equation calls for "thickness", not height). This is more practical
    # than taking the absolute value of the topography.
    density[oceans] = -1 * (density_water - density_crust)
    bouguer = 1e5 * 2 * np.pi * GRAVITATIONAL_CONST * density * topography
    return bouguer
