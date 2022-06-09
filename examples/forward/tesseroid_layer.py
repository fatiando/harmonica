"""
Layer of tesseroids
===================
"""
import boule as bl
import ensaio
import numpy as np
import verde as vd
import xarray as xr

import harmonica as hm

fname = ensaio.fetch_earth_topography(version=1)
topo = xr.load_dataarray(fname)
# print(topo)

region = (-75, -50, -55, -20)
topo_arg = topo.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))
# print(topo_arg)

ellipsoid = bl.WGS84

_, latitude_2d = np.meshgrid(topo_arg.longitude, topo_arg.latitude)
reference = ellipsoid.geocentric_radius(latitude_2d)
surface = topo_arg + reference
density = xr.where(topo_arg > 0, 2670, 2670 - 1040)

# print(surface)

tesseroids = hm.tesseroid_layer(
    coordinates=(topo_arg.longitude, topo_arg.latitude),
    surface=surface,
    reference=reference,
    properties={"density": density},
)
