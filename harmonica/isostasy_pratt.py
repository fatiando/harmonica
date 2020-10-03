"""
Function to calculate the densities of the roots assuming the
Pratt isostatic hypothesis.
"""
import numpy as np
import xarray as xr


def isostasy_pratt(
    topography,
    comp_depth=100e3
    density_crust = 2.8e3
    density_water=1e3,
    reference_depth=30e3,
):
    r"""
    Calculate the isostatic density from topography using Pratts's hypothesis.

 The Pratt hypothesis, developed by John Henry Pratt, English mathematician and Anglican 
 missionary, supposes that Earth’s crust has a uniform thickness below sea level with its 
 base everywhere supporting an equal weight per unit area at a depth of compensation. 
 In essence, this says that areas of the Earth of lesser density, such as mountain ranges, 
 project higher above sea level than do those of greater density. 
 The explanation for this was that the mountains resulted from the upward expansion of locally
 heated crustal material, which had a larger volume but a lower density after it had cooled.
 
 

        *Schematic of isostatic compensation following the Pratt hypothesis.*


    On the continents (positive topographic heights):

    .. math ::

      rho_{l} = \frac{comp_depth}{h+comp_depth} * rho_{c}

    while on the oceans (negative topographic heights):

    .. math ::
        rho_{o} = \frac{\rho_{c}*comp_depth - \rho_{w}*d}{comp_depth-d}

    in which :math:`d` is the bathymetry, :math:`h` is the topography, :math:`\rho_{l}` is the
    density of the mountain root, :math:`\rho_w` is the density of the water, and
    :math:`\rho_{c}` is the density of the crust, :math:`\rho_o` is the density of the 
    ocean root.  thickr is the constant root thickness that Pratt assumes 

    The computed densities will be added to the given reference Moho
    density (:math:`rho_{x}`).

    Parameters
    ----------
    topography : array or :class:`xarray.DataArray`
        Topography height and bathymetry depth in meters. It is usually prudent
        to use floating point values instead of integers to avoid integer
        division errors.
    density_crust : float
        Density of the crust in :math:`kg/m^3`.
    density_water : float
        Water density in :math:`kg/m^3`.
   compensation_depth : float
        The reference Moho depth (:math:`H`) in meters.

    Returns
    -------
    moho_rho : array or :class:`xarray.DataArray`
         The isostatic Moho density in meters.

    """
    
#Start here to formulate PRATT , Below is AIRY 
    # Need to cast to array to make sure numpy indexing works as expected for
    # 1D DataArray topography
    oceans = np.array(topography < 0)
    continent = np.logical_not(oceans)
    scale = np.full(topography.shape, np.nan, dtype="float")
    scale[continent] = density_crust * comp_depth / (continent + comp_depth)
    scale[oceans] = (density_crust*comp+depth - density_water*oceans)/ (comp_depth-oceans)
    moho = topography * scale + reference_depth
    if isinstance(moho, xr.DataArray):
        moho.name = "moho_density"
        moho.attrs["isostasy"] = "Pratt"
        moho.attrs["density_crust"] = str(density_crust)
        moho.attrs["comp_depth"] = str(comp_depth)
        moho.attrs["density_water"] = str(density_water)
        moho.attrs["reference_depth"] = str(reference_depth)
    return moho
