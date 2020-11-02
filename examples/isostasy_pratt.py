 """
 Pratt Isostasy
 =============
 
 The Pratt hypothesis, developed by John Henry Pratt, English mathematician, supposes that Earth’s crust has a uniform thickness below sea level with its 
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

    The computed densities will be added to the given reference
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
    rhot : array or :class:`xarray.DataArray`
         The isostatic density in  kg/m3.

    
    
Well use our sample topography data
(:func:`harmonica.datasets.fetch_topography_earth`) to calculate the Pratt isostatic
densities
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

# Calculate the isostatic density of the prisms above the reference depth of Moho
# Calculation for both ocean and continents 
rhot = hm.isostasy_pratt(data_africa.topography)
print("\nDensity grid:")
print(rhot)

# Draw the maps
plt.figure(figsize=(8, 9.5))
ax = plt.axes(projection=ccrs.LambertCylindrical(central_longitude=20))
pc = rhot.plot.pcolormesh(
    ax=ax, cmap="viridis_r", add_colorbar=False, transform=ccrs.PlateCarree()
)
plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.01, aspect=50, label="kg/m3")
ax.coastlines()
ax.set_title("Pratt isostatic density of prisms of Africa")
ax.set_extent(region, crs=ccrs.PlateCarree())
plt.show()





