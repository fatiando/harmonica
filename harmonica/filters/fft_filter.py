# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
FFT Based 2D Filter to potential field data
"""
import xrft
import numpy as np

class FFT_Filter:
    
    """
    Transform 2D potential field data use FFT based filter
    
    Workflow: Pad Data --> FFT --> Define and Apply Filter --> Inverse FFT --> Unpad Data
    

    Parameters
    ----------
    da: 2d-xarray
        This is the input 2D grid. Note, the input grid should be cartesian grid (northing, easting). 
        Geography grid (latitude,longitude) is invalid in here. 
        
    pad_width: None or list of arrays
        List containing the pad width [northing_pad_width, easting_pad_width] called by xrft.padding.pad. Default value is None,
        no padding apply to data.
        
    mode: str, default: "constant"
        str for control xrft.padding.pad function, One of the following string values (taken from numpy docs).
        - constant: Pads with a constant value.
        - edge: Pads with the edge values of array.
        - linear_ramp: Pads with the linear ramp between end_value and the
          array edge value.
        - maximum: Pads with the maximum value of all or part of the
          vector along each axis.
        - mean: Pads with the mean value of all or part of the
          vector along each axis.
        - median: Pads with the median value of all or part of the
          vector along each axis.
        - minimum: Pads with the minimum value of all or part of the
          vector along each axis.
        - reflect: Pads with the reflection of the vector mirrored on
          the first and last values of the vector along each axis.
        - symmetric: Pads with the reflection of the vector mirrored
          along the edge of the array.
        - wrap: Pads with the wrap of the vector along the axis.
          The first values are used to pad the end and the
          end values are used to pad the beginning.             
    
    """

    def __init__(self,da,pad_width=None,mode='constant',**kwargs):
        
        self.pad_width=pad_width
        self.da=da
        
        # Drop bad_coords
        for d in self.da.dims:
            bad_coords = [
                cname for cname in self.da.coords if cname != d and d in self.da[cname].dims
            ]
        for d in bad_coords:
            da=self.da.drop(bad_coords)

        # Pad Data
        if pad_width is None :
            da_padded = da
        else :
            da_padded = xrft.pad(da,{da.dims[0]:self.pad_width[0],da.dims[1]:self.pad_width[1]},mode)        
        
        # FFT
        da_fft = xrft.fft(da_padded,true_phase=True, true_amplitude=True)
        
        self.da_fft=da_fft

        # Frequency grids
        f_e, f_n = np.meshgrid(da_fft.coords[da_fft.dims[1]]*2*np.pi, da_fft.coords[da_fft.dims[0]]*2*np.pi)
        
        self.f_n=f_n
        self.f_e=f_e
    
    def fft(self):
        
        """
        Return forward fft result
    
        Returns
        -------
        da_fft : 2d-xarray
        """
        return self.da_fft

    def freq(self):
        
        """
        Return frequency grid [f_n,f_e]

        Returns
        -------
        f_n : 2d-array
        f_e : 2d-array
    
        """
        return self.f_n,self.f_e        
        
    def apply_filter(self,filter):
        
        """
        Apply filter and calculate Inverse FFT. Un_pad data if
        pad_with is not None.
        
        Parameters
        -------
        filter : 2d-array
            Apply pre-defined or custom filter 

        Returns
        -------
        da_unpad : 2d-xarray
            Xarray data in space domain after filtering.
        """       
         # Apply Filter
        da_filter = self.da_fft*filter
        # Inverse FFT
        da_ifft = xrft.ifft(da_filter,true_phase=True, true_amplitude=True)
        # Unpad Data
        if self.pad_width is None :
            da_unpad = da_ifft.real
        else :
            da_unpad=xrft.unpad(da_ifft.real,{da_ifft.dims[0]:self.pad_width[0],da_ifft.dims[1]:self.pad_width[1]}) 
            
        return da_unpad
    
    """
    -------
    Pre-defined filter lists as follows
    
    """        
    def derivative_e(self,order,savefilter=False,**kwargs):
        
        """
        Calculate n order horizental derivative along easting
        
        Parameters
        -------
        order : int
            The order of horizental derivative.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """     
            
        filter=np.power(self.f_e*1j,order)
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter

    def derivative_n(self,order,savefilter=False,**kwargs):
        """
        Calculate n order horizental derivative along northing
        
        Parameters
        -------
        order : int
            The order of horizental derivative.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """             
        filter=np.power(self.f_n*1j,order)
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter
        
    def derivative_v(self,order,savefilter=False,**kwargs):
        """
        Calculate n order vertical derivative
        
        Parameters
        -------
        order : int
            The order of vertical derivative.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """  
        
        filter=np.power(np.sqrt(self.f_e**2+self.f_n**2),order)
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter

    def guassian_lp(self,wavelength,savefilter=False,**kwargs):
        
        """
        Filter data by Guassian Low-pass filter 
        
        Parameters
        -------
        wavelength : float
            The cut off wavelength for low-pass filter. It has same unit with input xarray data.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """  
        
        filter=np.exp(-(self.f_e**2+self.f_n**2)/(2*(2*np.pi/wavelength)**2))
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter    

    def guassian_hp(self,wavelength,savefilter=False,**kwargs):

        """
        Filter data by Guassian High-pass filter 
        
        Parameters
        -------
        wavelength : float
            The cut off wavelength for high-pass filter. It has same unit with input xarray data.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """  
        
        filter=1-np.exp(-(self.f_e**2+self.f_n**2)/(2*(2*np.pi/wavelength)**2))
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter 
        
    def upward_continuation(self,height,savefilter=False,**kwargs):

        """
        Upward continuation data by height 
        
        Parameters
        -------
        height : float
            Heigh for upward continuation. The value should be negtive. It has same unit with input xarray data.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """  
        
        filter=np.exp(np.sqrt(self.f_e**2+self.f_n**2)*height)
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter
        
        
    def vertical_intergral(self,order=-1,savefilter=False,**kwargs):

        """
        Vertical intergral of potential field data 
        
        Parameters
        -------
        order : -1
            Vertical intergral. eg: transform gravity to gravity potential.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.
            
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """  
        
        filter=np.power(np.sqrt(self.f_e**2+self.f_n**2),order)

        filter=np.nan_to_num(filter,posinf=1,nan=1)

        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter
        
    def rtp(self,I,D,Im=None,Dm=None,savefilter=False,**kwargs):
        
        """
        Reduce total field magnetic anomaly data to the pole. For low inclination
        area, RTP is not stable. Recommod reduce total field magnetic anomaly data
        to the equator (rte).
        
        Parameters
        -------
        I : float in degree
            The inclination inducing Geomagnetic field.
            
        D : float in degree
            The declination inducing Geomagnetic field. 
            
        Im : float in degree
            The inclination of the total magnetization of the anomaly source. Default is I,
            neglecting remanent magnetization and self demagnetization.
            
        Dm : float in degree
            The declination of the total magnetization of the anomaly source. Default is D,
            neglecting remanent magnetization and self demagnetization.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.

        North
        
        ^                      --------> Horizental
        |   /                  |\ I
        |D /                   | \
        | /                    |  \
        |/                     |   \
        --------> East         Up
        
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """  
        # Transform degree to rad
        [I,D]=np.deg2rad([I,D])
        
        if Dm is None or Im is None:
            [Im,Dm]=[I,D]
        else:
            [Im,Dm]=np.deg2rad([Im,Dm])
        
        filter=(self.f_n**2+self.f_e**2)/((1j*(np.cos(I)*np.sin(D)*self.f_e+np.cos(I)*np.cos(D)*self.f_n)+
                                           np.sin(I)*np.sqrt(self.f_n**2+self.f_e**2))*
                                          (1j*(np.cos(Im)*np.sin(Dm)*self.f_e+np.cos(Im)*np.cos(Dm)*self.f_n)+
                                           np.sin(Im)*np.sqrt(self.f_n**2+self.f_e**2)))
        # Deal with inf and nan value
        filter=np.nan_to_num(filter,posinf=0,nan=0)

        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter     

    def rte(self,I,D,Im=None,Dm=None,savefilter=False,**kwargs):
        
        """
        Reduce total field magnetic anomaly data to the equator
        
        Parameters
        -------
        I : float in degree
            The inclination inducing Geomagnetic field.
            
        D : float in degree
            The declination inducing Geomagnetic field. 
            
        Im : float in degree
            The inclination of the total magnetization of the anomaly source. Default is I,
            neglecting remanent magnetization and self demagnetization.
            
        Dm : float in degree
            The declination of the total magnetization of the anomaly source. Default is D,
            neglecting remanent magnetization and self demagnetization.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.

        North
        
        ^                      --------> Horizental
        |   /                  |\ I
        |D /                   | \
        | /                    |  \
        |/                     |   \
        --------> East         Up
        
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """   
        
        # Transform degree to rad   
        [I,D]=np.deg2rad([I,D])
        
        if Dm is None or Im is None:
            [Im,Dm]=[I,D]
        else:
            [Im,Dm]=np.deg2rad([Im,Dm])
            

        filter=((1j*np.sin(D)*self.f_e+1j*np.cos(D)*self.f_n)**2)/((1j*(np.cos(I)*np.sin(D)*self.f_e+np.cos(I)*np.cos(D)*self.f_n)+
                                    np.sin(I)*np.sqrt(self.f_n**2+self.f_e**2))*
                                   (1j*(np.cos(Im)*np.sin(Dm)*self.f_e+np.cos(Im)*np.cos(Dm)*self.f_n)+
                                    np.sin(Im)*np.sqrt(self.f_n**2+self.f_e**2)))
        
        filter=np.nan_to_num(filter,posinf=0,nan=0)

        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            return da_out
        else:
            return filter     
        
    def pseudo_gravity(self,I,D,Im=None,Dm=None,F=50000,savefilter=False,**kwargs):

        """
        Pseudo gravity of total field magnetic anomaly data
        
        Parameters
        -------
        I : float in degree
            The inclination inducing Geomagnetic field.
            
        D : float in degree
            The declination inducing Geomagnetic field. 
            
        Im : float in degree
            The inclination of the total magnetization of the anomaly source. Default is I,
            neglecting remanent magnetization and self demagnetization.
            
        Dm : float in degree
            The declination of the total magnetization of the anomaly source. Default is D,
            neglecting remanent magnetization and self demagnetization.
            
        F : float or 2d-array
            Ambient field in the study area. It can use the mean ambinent field value in the study
            area or the real ambient field value in all locations. Default is 50,000 nT.
            
        savefilter : False or True
            If savefilter is False, direct apply filter to data, output in space domain.
            If savefilter is True, don't apply filter to data, output is the filter itself.

        North
        
        ^                      --------> Horizental
        |   /                  |\ I
        |D /                   | \
        | /                    |  \
        |/                     |   \
        --------> East         Up
        
        Returns
        -------
        da_out : 2d-xarray
            Xarray data in space domain after apply filter. Need savefilter = False.
            
        filter : 2d-array
            filter itself. Need savefilter = True.
        """   
        # Call rtp and vertical_intergral filter
        filter=self.rtp(I,D,Im,Dm,savefilter=True)*self.vertical_intergral(order=-1,savefilter=True)

        filter=np.nan_to_num(filter,posinf=0,nan=0)
        
        if savefilter is False :
            # Apply Filter
            da_out = self.apply_filter(filter)
            # Scale data by Ambient Field
            da_out = da_out/149.8/F
            return da_out
        else:
            return filter/149.8/F
        
