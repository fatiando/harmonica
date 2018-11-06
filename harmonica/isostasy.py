"""
Function to calculate the thickness of the roots and antiroots assuming the Airy isostatic hypothesis.
"""
import numpy as np


def isostasy_airy(topography, density_upper_crust, density_lower_crust,
                  density_mantle, density_water=None):
    """
    Computes the thickness of the roots and antiroots using the Airy hypothesis
    [Hofmann-WellenhofMoritz2006]_ .

    On continental points:

    .. math ::
        r = \frac{\rho_{uc}}{\rho_m - \rho_{lc}} t

    On oceanic points:

    .. math ::
        ar = \frac{\rho_{lc} - \rho_w}{\rho_m - \rho_{lc}} b

    where $t$ is the topography, $b$ is the bathymetry, $rho_m$ is the density of the mantle, $rho_w$ is the
    density of the water and $\rho_{uc}$ and $\rho_{lc}$ are the density of the
    upper and lower crust respectively. If $T$ is the normal thickness of the
    Earh's crust, $T + r$ and $T + ar$ are the isostatic Moho at the cotinental
    and oceanic points respectively.


    Parameters
    ----------
    density_mantle : float
        Mantle density in kg/m³.
    density_upper_crust : float
        Density of the upper crust in kg/m³.
    density_lower_crust : float
        Density of the lower crust in kg/m³.
    density_water : float
        Water density in kg/m³.
    topography : array
        Topography height and bathymetry depth in meters.

    Returns
    -------
    root : array
         Thickness of the roots and antiroot in meters.
    """
    root= topography.copy()

    root[topography >= 0] *= density_upper_crust / (density_mantle - density_lower_crust)

    if density_water is None:
        root[topography < 0] = np.nan
    else:
        root[topography < 0] *= (density_lower_crust - density_water) / (
            density_mantle - density_lower_crust
        )

    return root
