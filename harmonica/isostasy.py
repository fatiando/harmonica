"""
Function to calculate the thickness of the roots and antiroots assuming the
Airy isostatic hypothesis.
"""
import numpy as np


def isostasy_airy(
    topography,
    density_upper_crust,
    density_lower_crust,
    density_mantle,
    density_oceanic_crust=None,
    density_water=None,
):
    """
    Computes the thickness of the roots/antiroots using Airy's hypothesis.

    In Airy's hypothesis of isotasy, the mountain range can be thought of as a
    block of lithosphere (crust) floating in the asthenosphere. Mountains have
    roots (:math:`r`), while ocean basins have antiroots (:math:`ar`)
    [Hofmann-WellenhofMoritz2006]_ .
    If :math:`T` is the normal thickness of the Earh's crust, :math:`T + r` and
    :math:`T + ar` are the isostatic Moho at the cotinental and oceanic
    points respectively.

    On continental points:

    .. math ::
        r = \frac{\rho_{uc}}{\rho_m - \rho_{lc}} h

    On oceanic points:

    .. math ::
        ar = \frac{\rho_{oc} - \rho_w}{\rho_m - \rho_{oc}} h

    where :math:`h` is the topography/bathymetry, :math:`\rho_m` is the
    density of the mantle, :math:`\rho_w` is the density of the water,
    :math:`\rho_{oc}` is the density of the oceanic crust,:math:`\rho_{uc}` is
    the density of the upper crust and :math:`\rho_{lc}` is the density of the
    lower crust.

    Parameters
    ----------
    density_mantle : float
        Mantle density in :math:`kg/m^3`.
    density_upper_crust : float
        Density of the upper crust in :math:`kg/m^3`.
    density_lower_crust : float
        Density of the lower crust in :math:`kg/m^3`.
    density_oceanic_crust : float
        Density of the oceanic crust in :math:`kg/m^3`.
    density_water : float
        Water density in :math:`kg/m^3`.
    topography : array
        Topography height and bathymetry depth in meters.

    Returns
    -------
    root : array
         Thickness of the roots and antiroot in meters.
    """
    root = topography.copy()
    root[topography >= 0] *= density_upper_crust / (
        density_mantle - density_lower_crust
    )
    if density_water is None or density_oceanic_crust is None:
        root[topography < 0] = np.nan
    else:
        root[topography < 0] *= (density_oceanic_crust - density_water) / (
            density_mantle - density_oceani:wc_crust
        )
    return root
