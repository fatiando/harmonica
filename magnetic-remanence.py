import numpy as np
import verde as vd
import matplotlib.pyplot as plt

from harmonica import (
    OblateEllipsoid,
    ProlateEllipsoid,
    TriaxialEllipsoid,
    ellipsoid_magnetics,
)

external_field = (55_000, 48, -12)
susceptibility = 0.5

region = (-200, 200, -200, 200)
shape = (151, 151)
height = 100.0
coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)


center = (0, 0, 0)
yaw, pitch = 90, 0
ellipsoid = ProlateEllipsoid(
    a=50,
    b=49,
    yaw=yaw,
    pitch=pitch,
    centre=center,
)
rem_magnetization = (1, 0, 0)

be, bn, bu = ellipsoid_magnetics(
    coordinates,
    ellipsoid,
    susceptibilities=0,
    external_field=external_field,
    remnant_mag=rem_magnetization,
)


# Plot
# ----
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 10), sharex=True, sharey=True)
for ax, b_component in zip(axes, (be, bn, bu), strict=True):
    tmp = ax.pcolormesh(*coordinates[:2], b_component)
    ax.set_aspect("equal")
    plt.colorbar(tmp, ax=ax)

plt.show()
