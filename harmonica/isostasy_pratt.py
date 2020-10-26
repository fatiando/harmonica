"""
Function to calculate the densities of the roots assuming the
Pratt isostatic hypothesis.
"""
import numpy as np
import xarray as xr


def isostasy_pratt(
    topography,
    comp_depth=100e3,
    density_crust=2.8e3,
    density_water=1e3,
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
    scale = np.full(topography.shape, np.nan, dtype="float") #makes empty arrays of scale
    continent_v = topography*continent #only values where continents are
    oceans_v = topography*oceans*-1 #positive values 
    scale[continent_v] = density_crust * comp_depth / (continent_v + comp_depth) 
    scale[oceans_v] = (density_crust*comp_depth - density_water*oceans_v)/ (comp_depth-oceans_v)
    rhot = scale
    if isinstance(rhot, xr.DataArray):
        rhot.name = "moho_density"
        rhot.attrs["isostasy"] = "Pratt"
        rhot.attrs["density_crust"] = str(density_crust)
        rhot.attrs["comp_depth"] = str(comp_depth)
        rhot.attrs["density_water"] = str(density_water)
        rhot.attrs["reference_depth"] = str(reference_depth)
    return rhot


"TESTING WITHOUT NAMING FUNCTION TO SEE IF ALL PARTS WORK

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the elevation model and cut out the portion of the data corresponding to
# Africa
data = hm.datasets.fetch_topography_earth()
region = (-20, 60, -40, 45)
data_africa = data.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))

topography=data_africa.topography
comp_depth=100e3
density_crust=2.67e3
density_water=1e3

   # Need to cast to array to make sure numpy indexing works as expected for
    # 1D DataArray topography
    oceans = np.array(topography < 0)
    continent = np.logical_not(oceans)
    scale = np.full(topography.shape, np.nan, dtype="float") #makes empty arrays of scale
    continents_v = topography*continent #only values where continents are
    oceans_v = topography*oceans*-1 #positive values where ocean values are
    scale[continent_v] = density_crust * comp_depth / (continent_v + comp_depth)
    scale[oceans_v] = (density_crust*comp_depth - density_water*oceans_v)/ (comp_depth-oceans_v)
    rhot = cont+oce
    if isinstance(rhot, xr.DataArray):
        rhot.name = "moho_density"
        rhot.attrs["isostasy"] = "Pratt"
        rhot.attrs["density_crust"] = str(density_crust)
        rhot.attrs["comp_depth"] = str(comp_depth)
        rhot.attrs["density_water"] = str(density_water)


plt.figure(figsize=(8, 9.5))
ax = scale.axes(projection=ccrs.LambertCylindrical(central_longitude=20))
pc = continent_v.plot.pcolormesh(
    ax=ax, cmap="viridis_r", add_colorbar=True, transform=ccrs.PlateCarree()
)
plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.01, aspect=50, label="rho kg/m3")
ax.coastlines()
ax.set_title("pratt isostatic density of Africa")
ax.set_extent(region, crs=ccrs.PlateCarree())
plt.show()
plt.savefig('pratt.jpg')




