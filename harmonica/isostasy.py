"""
Function to calculate the thickness of the roots and antiroots assuming the
Airy isostatic hypothesis.
"""
import numpy as np


def isostasy_airy(
    topography,
    density_crust=2.8e3,
    density_mantle=3.3e3,
    density_water=1e3,
    reference_depth=30e3,
):
    r"""
    Calculate the isostatic Moho depth from topography using Airy's hypothesis.

    According to the Airy hypothesis of isostasy, topography above sea level is
    supported by a thickening of the crust (a root) while oceanic basins are supported
    by a thinning of the crust (an anti-root).

    .. figure:: ../../_static/figures/airy-isostasy.svg
        :align: center
        :width: 400px

        *Schematic of isostatic compensation following the Airy hypothesis.*

    The relationship between the topographic/bathymetric heights (:math:`h`) and the
    root thickness (:math:`r`) is governed by mass balance relations and can be found in
    classic textbooks like [TurcotteSchubert2014]_ and [Hofmann-WellenhofMoritz2006]_.

    On the continents (positive topographic heights):

    .. math ::

        r = \frac{\rho_{c}}{\rho_m - \rho_{c}} h

    while on the oceans (negative topographic heights):

    .. math ::
        r = \frac{\rho_{c} - \rho_w}{\rho_m - \rho_{c}} h

    in which :math:`h` is the topography/bathymetry, :math:`\rho_m` is the density of
    the mantle, :math:`\rho_w` is the density of the water, and :math:`\rho_{c}` is the
    density of the crust.

    The computed root thicknesses will be added to the given reference Moho depth
    (:math:`H`) to arrive at the isostatic Moho depth. Use ``reference_depth=0`` if you
    want the values of the root thicknesses instead.

    Parameters
    ----------
    topography : array or :class:`xarray.DataArray`
        Topography height and bathymetry depth in meters.
    density_crust : float
        Density of the crust in :math:`kg/m^3`.
    density_mantle : float
        Mantle density in :math:`kg/m^3`.
    density_water : float
        Water density in :math:`kg/m^3`.
    reference_depth : float
        The reference Moho depth (:math:`H`) in meters.

    Returns
    -------
    moho_depth : array or :class:`xarray.DataArray`
         The isostatic Moho depth in meters.

    """
    root = topography.astype(np.float64)
    root[topography >= 0] *= density_crust / (
        density_mantle - density_crust
    )
    root[topography < 0] *= (density_crust - density_water) / (
        density_mantle - density_crust
    )
    return root
