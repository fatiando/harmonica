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
    scale = np.full(topography.shape, np.nan, dtype="float") #makes empty arrays of scale
    continent_v = topography*continent
    oceans_v = topography*oceans
    cont = density_crust * comp_depth / (continent_v + comp_depth) #not sure about the continent variable
    oce = (density_crust*comp_depth - density_water*oceans_v)/ (comp_depth-oceans_v)
    mohod = cont+oce
    if isinstance(mohod, xr.DataArray):
        moho.name = "moho_density"
        moho.attrs["isostasy"] = "Pratt"
        moho.attrs["density_crust"] = str(density_crust)
        moho.attrs["comp_depth"] = str(comp_depth)
        moho.attrs["density_water"] = str(density_water)
        moho.attrs["reference_depth"] = str(reference_depth)
    return mohod

##TESTING 


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the elevation model and cut out the portion of the data corresponding to
# Africa
data = hm.datasets.fetch_topography_earth()
region = (-20, 60, -40, 45)
data_africa = data.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))
print("Topography/bathymetry grid:")
print(data_africa)

# Calculate the isostatic Moho depth using the default values for densities and
# reference Moho
moho = isostasy_pratt(data_africa.topography)
print("\nMoho depth grid:")
print(moho)

# Draw the maps
plt.figure(figsize=(8, 9.5))
ax = plt.axes(projection=ccrs.LambertCylindrical(central_longitude=20))
pc = moho.plot.pcolormesh(
    ax=ax, cmap="viridis_r", add_colorbar=True, transform=ccrs.PlateCarree()
)
plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.01, aspect=50, label="meters")
ax.coastlines()
ax.set_title("Airy isostatic Moho depth of Africa")
ax.set_extent(region, crs=ccrs.PlateCarree())
plt.show()
plt.savefig('airy.jpg')


"""
Airy Isostasy
=============

According to the Airy hypothesis of isostasy, topography above sea level is
supported by a thickening of the crust (a root) while oceanic basins are
supported by a thinning of the crust (an anti-root). Function
:func:`harmonica.isostasy_airy` computes the depth to crust-mantle interface
(the Moho) according to Airy isostasy. One must assume a value for the
reference thickness of the continental crust in order to convert the
root/anti-root thickness into Moho depth. The function contains common default
values for the reference thickness and crust, mantle, and water densities
[TurcotteSchubert2014]_.

We'll use our sample topography data
(:func:`harmonica.datasets.fetch_topography_earth`) to calculate the Airy
isostatic Moho depth of Africa.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import harmonica as hm

# Load the elevation model and cut out the portion of the data corresponding to
# Africa
data = hm.datasets.fetch_topography_earth()
region = (-20, 60, -40, 45)
data_africa = data.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))
print("Topography/bathymetry grid:")
print(data_africa)

# Calculate the isostatic Moho depth using the default values for densities and
# reference Moho
moho = hm.isostasy_airy(data_africa.topography)
print("\nMoho depth grid:")
print(moho)

# Draw the maps
plt.figure(figsize=(8, 9.5))
ax = plt.axes(projection=ccrs.LambertCylindrical(central_longitude=20))
pc = moho.plot.pcolormesh(
    ax=ax, cmap="viridis_r", add_colorbar=True, transform=ccrs.PlateCarree()
)
plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.01, aspect=50, label="meters")
ax.coastlines()
ax.set_title("Airy isostatic Moho depth of Africa")
ax.set_extent(region, crs=ccrs.PlateCarree())
plt.show()
plt.savefig('airy.jpg')


