"""
Function to calculate the moho undulation assuming the Airy isostatic hypothesis.
"""
import numpy as np


def isostasy_airy(topography, density_crust, density_mantle, density_water=None):
    """
    Computes the Moho undulation from topography using the Airy hypothesis.

    On continental points:

    .. math ::
        m = \frac{\rho_c}{\rho_m - \rho_c} h

    On oceanic points:

    .. math ::
        m = \frac{\rho_c - \rho_w}{\rho_m - \rho_c} h

    where $L$ is the topography, $rho_m$ is the density of the mantle,
    $rho_w$ is the density of the water, $\rho_c$ is the crustal density.

    Parameters
    ----------
    density_mantle : float
        Mantle density in kg/m³.
    density_crust : float
        Crustal density in kg/m³.
    density_water : float
        Water density in kg/m³.
    topography : array
        Topography height and bathymetry depth in meters.

    Returns
    -------
    moho_undulation : array
        Isostatic moho undulation in meters.
    """
    moho_undulation = topography.copy()

    moho_undulation[topography >= 0] *= density_crust / (density_mantle - density_crust)

    if density_water is None:
        moho_undulation[topography < 0] = np.nan
    else:
        moho_undulation[topography < 0] *= (density_crust - density_water) / (
            density_mantle - density_crust
        )

    return moho_undulation
