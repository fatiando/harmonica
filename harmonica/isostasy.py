import numpy as np


def isostasy_airy(topography, density_crust, density_mantle, density_water=None):
    """
    Computes the moho undulation using the Airy hypothesis.

    On continental points:

    .. math ::
        m = \frac{\rho_c}{\rho_m - \rho_c} h

    On oceanic points:

    .. math ::
        m = \frac{\rho_c - \rho_w}{\rho_m - \rho_c} h

    where $L$ is the topography, $rho_m$ is the density of the mantle,
    $rho_w$ is the density of the water, $\rho_c$ is the crustal density.

    Parameters:

    * rho_m: float
        Mantle density in kg/m$^3$.

    * rho_c: float
        Crustal density in kg/m$^3$.

    * rho_w: float
        Water density in kg/m$^3$.

    * h: float
        Topography in meters.

    moho:

    * m: 1D-array
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
