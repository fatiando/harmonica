# Define a set of prisms and their densities:
#
prisms = [
    [0, 4, 0, 5, -10, 0],
    [0, 4, 7, 9, -12, -3],
    [6, 9, 2, 6, -7, 3],
]
densities = [2900, 3000, 2670]
#
# Generate a :class:`pyvista.UnstructuredGrid` out of them:
#
import harmonica as hm
pv_grid = hm.visualization.prism_to_pyvista(
    prisms, properties={"density": densities}
)
pv_grid # doctest: +SKIP
#
# Plot it using :mod:`pyvista`:
#
pv_grid.plot() # doctest: +SKIP
