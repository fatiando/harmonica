# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Function to calculate the thickness of the roots and antiroots assuming the
Airy isostatic hypothesis.
"""
import xarray as xr


def isostasy_airy(
    basement_elevation,
    layer_thickness=(),
    layer_density=(),
    density_crust=2.8e3,
    density_mantle=3.3e3,
    reference_depth=30e3,
):
    r"""
    Calculate the isostatic Moho depth from rock equvalent topography using
    Airy's hypothesis.

    According to the Airy hypothesis of isostasy, rock equvalent topography
    above sea level is supported by a thickening of the crust (a root) while
    rock equvalent topography below sea level is supported by a thinning of
    the crust (an anti-root). This assumption is usually

    .. figure:: ../../_static/figures/airy-isostasy.png
        :align: center
        :width: 400px

        *Schematic of isostatic compensation following the Airy hypothesis.*

    The relationship between the rock equvalent topography (:math:`ret`)
    and the root thickness (:math:`r`) is governed by mass balance relations
    and can be found in classic textbooks like [TurcotteSchubert2014]_ and
    [Hofmann-WellenhofMoritz2006]_.

    Compress all layers's mass above basement (:math:`h`) into rock equvalent
    topography [Balmino_etal1973]_ :

    .. math ::

        ret = h + \frac{\rho_{i}}{\rho_{c}}th_{i} + ..

    Based on rock equvalent topography, the root is calculated as:

    .. math ::
        r = \frac{\rho_{c}}{\rho_m - \rho_{c}} ret

    in which :math:`ret` is the rock equvalent topography , :math:`\rho_m` is
    the density of the mantle, and :math:`\rho_{c}` is the density of the
    crust.

    The computed root thicknesses will be added to the given reference Moho
    depth (:math:`H`) to arrive at the isostatic Moho depth. Use
    ``reference_depth=0`` if you want the values of the root thicknesses
    instead.

    Parameters
    ----------
    basement_elevation : array or :class:`xarray.DataArray`
        Basement elevation in meters. It usually refer to topography height
        and bathymetry depth without sediment cover. When considering
        sedimentary layer, it refer to crystalline basement. It is usually
        prudent to use floating point values instead of integers to avoid
        integer division errors.
    layer_thickness : tuple contains `xarray.DataArray`
        Layer thickness in meters. It refer to all layers above
        topography/basement, including ice, water, and sediment.
    layer_density : tuple contains `xarray.DataArray` or float
        Layer density in :math:`kg/m^3`. It refer to all layers above
        topography/basement, including ice, water, and sediment.
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

    # Define scale factor to calculate Airy root
    scale = density_crust / (density_mantle - density_crust)

    # For the case: multi-layers above topography/basement
    if type(layer_thickness) is tuple:
        # Define total mass of layers abrove topography/basement
        layer_mass = tuple(
            ele1 * ele2 for ele1, ele2 in zip(layer_thickness, layer_density)
        )
        # Calculate equvalent topography
        rock_equavalent_topography = (
            basement_elevation + sum(list(layer_mass)) / density_crust
        )
    else:
        # For the case: a single layer or no layer above topography/basement
        layer_mass = layer_thickness * layer_density
        rock_equavalent_topography = basement_elevation + layer_mass / density_crust
    # Calculate Moho depth
    moho = rock_equavalent_topography * scale + reference_depth
    if isinstance(moho, xr.DataArray):
        moho.name = "moho_depth"
        moho.attrs["isostasy"] = "Airy"
        moho.attrs["density_crust"] = str(density_crust)
        moho.attrs["density_mantle"] = str(density_mantle)
    return moho
