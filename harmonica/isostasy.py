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
import warnings

import numpy as np
import xarray as xr


def isostatic_moho_airy(
    basement,
    layers=None,
    density_crust=2.8e3,
    density_mantle=3.3e3,
    reference_depth=30e3,
):
    r"""
    Calculate the isostatic Moho depth using Airy's hypothesis.

    Take the height of the crystalline basement and optional additional layers
    located on top of it. Each layer must be specified through its vertical
    thickness and its corresponding density. Return the depth to the
    Mohorovicic discontinuity (crust-mantle interface).

    Parameters
    ----------
    basement : float or array
        Height of the crystalline basement in meters.
        It usually refer to topography and bathymetry height without sediment
        cover.
        When considering sedimentary basins, it refers to crystalline basement
        (topography/bathymetry minus sediment thickness).
    layers : dict (optional)
        Dictionary that contains information about the thickness and density of
        the layers located above the ``basement``.
        For each layer, a single item should be created: its key will be the
        layer name as a ``str`` and its values must be tuples containing the
        layer thickness in meters and the layer density (in :math:`kg/m^3`)
        in that order.
        Thicknesses and densities can be floats or arrays.
        If ``None``, no layers will be considered.
        Default as ``None``.
    density_crust : float or array (optional)
        Density of the crust in :math:`kg/m^3`.
    density_mantle : float or array (optional)
        Mantle density in :math:`kg/m^3`.
    reference_depth : float or array (optional)
        The reference Moho depth (:math:`H`) in meters.

    Returns
    -------
    moho_depth : float or array
         The isostatic Moho depth in meters.

    Notes
    -----
    According to the Airy hypothesis of isostasy, rock equivalent topography
    above sea level is supported by a thickening of the crust (a root) while
    rock equivalent topography below sea level is supported by a thinning of
    the crust (an anti-root). This assumption is usually

    .. figure:: ../../_static/figures/airy-isostasy-moho.svg
        :align: center
        :width: 400px

        *Schematic of isostatic compensation following the Airy hypothesis.*

    The relationship between the rock equivalent topography (:math:`r_{et}`)
    and the root thickness (:math:`r`) is governed by mass balance relations
    and can be found in classic textbooks like [TurcotteSchubert2014]_ and
    [Hofmann-WellenhofMoritz2006]_.

    Compress all layers' mass above basement (:math:`h`) into *rock equivalent
    topography* [Balmino1973]_ :

    .. math ::

        r_{et} = h + \sum\limits_{i=1}^N \frac{\rho_{i}}{\rho_{c}} t_{i}

    Based on rock equivalent topography, the root is calculated as:

    .. math ::
        r = \frac{\rho_{c}}{\rho_m - \rho_{c}} r_{et}

    in which :math:`r_{et}` is the rock equivalent topography , :math:`\rho_m`
    is the density of the mantle, and :math:`\rho_{c}` is the density of the
    crust.

    The computed root thicknesses will be added to the given reference Moho
    depth (:math:`H`) to arrive at the isostatic Moho depth. Use
    ``reference_depth=0`` if you want the values of the root thicknesses
    instead.

    Examples
    --------
    Simple model of continental topography with a sedimentary basin on top

    >>> # Define crystalline basement height (in meters)
    >>> basement = 1200
    >>> # Define a layer of sediments with a thickness of 200m
    >>> sediments_thickness = 200
    >>> sediments_density = 2300
    >>> # Get depth of the Moho following Airy's isostatic hypothesis
    >>> moho_depth = isostatic_moho_airy(
    ...     basement,
    ...     layers={"sediments": (sediments_thickness, sediments_density)}
    ... )
    >>> moho_depth
    37640.0

    Simple model of oceanic sedimentary basin

    >>> # Define bathymetry (in meters)
    >>> bathymetry = -3000
    >>> # Define a layer of sediments with a thickness of 400m
    >>> sediments_thickness = 400
    >>> sediments_density = 2200
    >>> # Define a layer for the oceanic water
    >>> water_thickness = abs(bathymetry)
    >>> water_density = 1040
    >>> # Get depth of the Moho following Airy's isostatic hypothesis
    >>> moho_depth = isostatic_moho_airy(
    ...     bathymetry - sediments_thickness,
    ...     layers={
    ...         "sediments": (sediments_thickness, sediments_density),
    ...         "water": (water_thickness, water_density),
    ...     }
    ... )
    >>> moho_depth
    18960.0


    """
    # Compute equivalent topography for the layers (if any)
    layers_equivalent_topography = 0
    if layers is not None:
        for thickness, density in layers.values():
            layers_equivalent_topography += thickness * density
        layers_equivalent_topography /= density_crust

    # Calculate rock equivalent topography
    rock_equivalent_topography = basement + layers_equivalent_topography

    # Calculate Moho depth
    scale = density_crust / (density_mantle - density_crust)
    moho = rock_equivalent_topography * scale + reference_depth

    # Add attributes to the xr.DataArray
    if isinstance(moho, xr.DataArray):
        moho.name = "moho_depth"
        moho.attrs["isostasy"] = "Airy"
        moho.attrs["density_crust"] = str(density_crust)
        moho.attrs["density_mantle"] = str(density_mantle)
        if layers is not None:
            for name, (_, density) in layers.items():
                moho.attrs[f"density_{name}"] = density
    return moho


def isostasy_airy(
    topography,
    density_crust=2.8e3,
    density_mantle=3.3e3,
    density_water=1e3,
    reference_depth=30e3,
):
    r"""
    Calculate the isostatic Moho depth from topography using Airy's hypothesis.

    .. warning::

        The :func:`harmonica.isostasy_airy` function will be deprecated in
        Harmonica v0.6. Please use :func:`harmonica.isostatic_moho_airy`
        instead.

    According to the Airy hypothesis of isostasy, topography above sea level is
    supported by a thickening of the crust (a root) while oceanic basins are
    supported by a thinning of the crust (an anti-root). This assumption is
    usually

    .. figure:: ../../_static/figures/airy-isostasy.svg
        :align: center
        :width: 400px

        *Schematic of isostatic compensation following the Airy hypothesis.*

    The relationship between the topographic/bathymetric heights (:math:`h`)
    and the root thickness (:math:`r`) is governed by mass balance relations
    and can be found in classic textbooks like [TurcotteSchubert2014]_ and
    [Hofmann-WellenhofMoritz2006]_.

    On the continents (positive topographic heights):

    .. math ::

        r = \frac{\rho_{c}}{\rho_m - \rho_{c}} h

    while on the oceans (negative topographic heights):

    .. math ::
        r = \frac{\rho_{c} - \rho_w}{\rho_m - \rho_{c}} h

    in which :math:`h` is the topography/bathymetry, :math:`\rho_m` is the
    density of the mantle, :math:`\rho_w` is the density of the water, and
    :math:`\rho_{c}` is the density of the crust.

    The computed root thicknesses will be added to the given reference Moho
    depth (:math:`H`) to arrive at the isostatic Moho depth. Use
    ``reference_depth=0`` if you want the values of the root thicknesses
    instead.

    Parameters
    ----------
    topography : array or :class:`xarray.DataArray`
        Topography height and bathymetry depth in meters. It is usually prudent
        to use floating point values instead of integers to avoid integer
        division errors.
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
    warnings.warn(
        "The harmonica.isostasy_airy function will be deprecated in "
        + "Harmonica v0.6. Please use harmonica.isostatic_moho_airy "
        + "instead.",
        FutureWarning,
    )
    # Need to cast to array to make sure numpy indexing works as expected for
    # 1D DataArray topography
    oceans = np.array(topography < 0)
    continent = np.logical_not(oceans)
    scale = np.full(topography.shape, np.nan, dtype="float")
    scale[continent] = density_crust / (density_mantle - density_crust)
    scale[oceans] = (density_crust - density_water) / (density_mantle - density_crust)
    moho = topography * scale + reference_depth
    if isinstance(moho, xr.DataArray):
        moho.name = "moho_depth"
        moho.attrs["isostasy"] = "Airy"
        moho.attrs["density_crust"] = str(density_crust)
        moho.attrs["density_mantle"] = str(density_mantle)
        moho.attrs["density_water"] = str(density_water)
        moho.attrs["reference_depth"] = str(reference_depth)
    return moho
