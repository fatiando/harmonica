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
    layers=None,
    density_crust=2.8e3,
    density_mantle=3.3e3,
    reference_depth=30e3,
):
    r"""
    Calculate the isostatic Moho depth from rock equivalent topography using
    Airy's hypothesis.

    According to the Airy hypothesis of isostasy, rock equivalent topography
    above sea level is supported by a thickening of the crust (a root) while
    rock equivalent topography below sea level is supported by a thinning of
    the crust (an anti-root). This assumption is usually

    .. figure:: ../../_static/figures/airy-isostasy.png
        :align: center
        :width: 400px

        *Schematic of isostatic compensation following the Airy hypothesis.*

    The relationship between the rock equivalent topography (:math:`ret`)
    and the root thickness (:math:`r`) is governed by mass balance relations
    and can be found in classic textbooks like [TurcotteSchubert2014]_ and
    [Hofmann-WellenhofMoritz2006]_.

    Compress all layers's mass above basement (:math:`h`) into rock equivalent
    topography [Balmino_etal1973]_ :

    .. math ::

        ret = h + \frac{\rho_{i}}{\rho_{c}}th_{i} + ..

    Based on rock equivalent topography, the root is calculated as:

    .. math ::
        r = \frac{\rho_{c}}{\rho_m - \rho_{c}} ret

    in which :math:`ret` is the rock equivalent topography , :math:`\rho_m` is
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
        and bathymetry depth in area without sediment cover. When considering
        sedimentary layer, it refers to crystalline basement
        (topography/bathymetry minues sediment thickness). It is usually
        prudent to use floating point values instead of integers to avoid
        integer division errors.
    layers : dictionary contains tuples as {"names": (thickness , density)},
        default as None.
        Thickness and density type: float, array or :class:`xarray.DataArray`
        Layer thickness in meters. Layer density in :math:`kg/m^3`.It refer to
        all layers above basement, including ice, water, and sediment.
    density_crust : float
        Density of the crust in :math:`kg/m^3`.
    density_mantle : float
        Mantle density in :math:`kg/m^3`.
    reference_depth : float
        The reference Moho depth (:math:`H`) in meters.

    Returns
    -------
    moho_depth : array or :class:`xarray.DataArray`
         The isostatic Moho depth in meters.
    """

    # Define scale factor to calculate Airy root
    scale = density_crust / (density_mantle - density_crust)

    # Define initial mass
    mass_layers = 0
    name_layers = []
    density_layers = []

    # No mass load above basement
    if layers is None:
        name_layers = "None"
        density_layers = "None"
    # With mass load above basement
    else:
        # Calculate total mass above basement
        for sub_layer_name, sub_layer in layers.items():
            mass_layers += sub_layer[0] * sub_layer[1]
            name_layers.append(sub_layer_name)
            density_layers.append(str(sub_layer[1]))

    # Calculate rock equivalent topography
    rock_equivalent_topography = basement_elevation + mass_layers / density_crust

    # Calculate Moho depth
    moho = rock_equivalent_topography * scale + reference_depth
    if isinstance(moho, xr.DataArray):
        moho.name = "moho_depth"
        moho.attrs["isostasy"] = "Airy"
        moho.attrs["name_layers"] = name_layers
        moho.attrs["density_layers"] = density_layers
        moho.attrs["density_crust"] = str(density_crust)
        moho.attrs["density_mantle"] = str(density_mantle)
    return moho
