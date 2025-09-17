import numpy as np
import verde as vd
import matplotlib.pyplot as plt

from harmonica import (
    OblateEllipsoid,
    ProlateEllipsoid,
    TriaxialEllipsoid,
    ellipsoid_magnetics,
)
from functions.utils_ellipsoids import _sphere_magnetic, _get_sphere_magnetization


external_field = (55_000, 48, -12)
susceptibility = 0.5

region = (-200, 200, -200, 200)
shape = (151, 151)
height = 100.0
coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)

b_fields = {}

# Compute magnetic field of the sphere
radius = 50.0
center = (0, 0, 0)
magnetization = _get_sphere_magnetization(susceptibility, external_field)

b_fields["sphere"] = b_sphere = _sphere_magnetic(
    coordinates, radius, center, magnetization
)


# Compute magnetic field of ellipsoids that approximate a sphere
yaw, pitch, roll = 0, 0, 0
a = radius
delta = 0.1

oblate = OblateEllipsoid(
    a=a,
    b=a + delta,
    yaw=yaw,
    pitch=pitch,
    centre=center,
)
prolate = ProlateEllipsoid(
    a=a,
    b=a - delta,
    yaw=yaw,
    pitch=pitch,
    centre=center,
)
triaxial = TriaxialEllipsoid(
    a=a,
    b=a - delta,
    c=a - 2 * delta,
    yaw=yaw,
    pitch=pitch,
    roll=roll,
    centre=center,
)

ellipsoids = {
    "oblate": oblate,
    "prolate": prolate,
    "triaxial": triaxial,
}

for name, ellipsoid in ellipsoids.items():
    b_fields[name] = ellipsoid_magnetics(
        coordinates,
        ellipsoid,
        susceptibility,
        external_field,
    )


# Plot
# ----
fig, axes = plt.subplots(
    nrows=3, ncols=len(b_fields), figsize=(16, 10), sharex=True, sharey=True
)

for i, (name, b_field) in enumerate(b_fields.items()):
    axes_column = axes[:, i]
    for ax, b_component in zip(axes_column, b_field, strict=True):
        tmp = ax.pcolormesh(*coordinates[:2], b_component)
        ax.set_aspect("equal")
        plt.colorbar(tmp, ax=ax)
    axes_column[0].set_title(name)

plt.show()
