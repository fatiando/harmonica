import numpy as np
   
def isostasy_airy(topography, density_crust, density_mantle, 
                  density_water=None):
    """
        Computes the isostatic moho using the topography.

        On continental points:

        .. math ::
            moho_depth = \frac{\rho_c}{\rho_m - \rho_c} L

        On oceanic points:
        
        .. math ::
            moho_depth = \frac{\rho_c - \rho_w}{rho_m - rho_c} L

        where $L$ is the topography, $rho_m$ is the density of the mantle, 
        $rho_w$ is the density of the water, $\rho_c$ is the crustal density.

        Parameters:

        * rho_m: float
            Mantle density in kg/m$^3$.

        * rho_c: float
            Crustal density in kg/m$^3$.
            
        * rho_w: float
            Water density in kg/m$^3$.
            
        * L: float
            Topography in meters.
            
        moho_depth:
        
        * h: 1D-array
            Isostatic moho in meters.
        """
      
