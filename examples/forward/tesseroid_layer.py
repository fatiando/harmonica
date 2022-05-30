"""
Layer of tesseroids
===================
"""
import boule as bl
import ensaio
import numpy as np
import verde as vd
import xarray as xr
import numpy as np
import harmonica as hm

fname = ensaio.fetch_earth_topography(version=1)
topo = xr.load_dataarray(fname)
# print(topo)

region = (-70, -65, -40, -30)
topo_arg = topo.sel(latitude=slice(*region[2:]), longitude=slice(*region[:2]))
# print(topo_arg)

ellipsoid = bl.WGS84

surface = topo_arg + ellipsoid.mean_radius
density = 2670 * np.ones_like(surface)
# print(surface)

tesseroids = hm.tesseroid_layer(
    coordinates=(topo_arg.longitude, topo_arg.latitude),
    surface=surface,
    reference=ellipsoid.mean_radius,
    properties={"density": density},
)
print(tesseroids)
