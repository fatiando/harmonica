# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions for visualizing prisms through pyvista
"""
import numpy as np

try:
    import pyvista
except ImportError:
    pyvista = None
else:
    import vtk


def prism_to_pyvista(prisms, properties=None):
    """
    Create a :class:`pyvista.UnstructuredGrid` out of prisms

    Builds a :class:`pyvista.UnstructuredGrid` out of a set of prisms that
    could be used to plot a 3D representation through :mod:`pyvista`.

    Parameters
    ----------
    prisms : list or 2d-array
        List or 2d-array with the boundaries of the prisms.
        Each row contains the boundaries of each prism in the following order:
        ``west``, ``east``, ``south``, ``north``, ``bottom``, ``top``.
    properties : dict or None (optional)
        Dictionary with the physical properties of the prisms.
        Each key should be a string and its corresponding value a 1D array.
        If None, no property will be added to the
        :class:`pyvista.UnstructuredGrid`.
        Default to None.

    Returns
    -------
    pv_grid : :class:`pyvista.UnstructuredGrid`
        :class:`pyvista.UnstructuredGrid` that represents the prisms with their
        properties (if any).

    Examples
    --------

    .. pyvista-plot::

       Define a set of prisms and their densities:

       >>> prisms = [
       ...     [0, 4, 0, 5, -10, 0],
       ...     [0, 4, 7, 9, -12, -3],
       ...     [6, 9, 2, 6, -7, 3],
       ... ]
       >>> densities = [2900, 3000, 2670]

       Generate a :class:`pyvista.UnstructuredGrid` out of them:

       >>> import harmonica as hm
       >>> pv_grid = hm.visualization.prism_to_pyvista(
       ...     prisms, properties={"density": densities}
       ... )
       >>> pv_grid # doctest: +SKIP

       Plot it using :mod:`pyvista`:

       >>> pv_grid.plot() # doctest: +SKIP

    """
    # Check if pyvista are installed
    if pyvista is None:
        raise ImportError(
            "Missing optional dependency 'pyvista' required for building pyvista grids."
        )
    # Get prisms and number of prisms
    prisms = np.atleast_2d(prisms)
    n_prisms = prisms.shape[0]
    # Get the vertices of the prisms
    vertices = _prisms_boundaries_to_vertices(prisms)
    # Generate the cells for the hexahedrons
    cells = np.arange(n_prisms * 8).reshape([n_prisms, 8])
    # Build the UnstructuredGrid
    pv_grid = pyvista.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells}, vertices)
    # Add properties to the grid
    if properties is not None:
        for name, prop in properties.items():
            # Check if the property is given as 1d array
            prop = np.atleast_1d(prop)
            if prop.ndim > 1:
                raise ValueError(
                    f"Multidimensional array found in '{name}' property. "
                    + "Please, pass prism properties as 1d arrays."
                )
            # Assign the property to the cell_data
            pv_grid.cell_data[name] = prop
    return pv_grid


def _prisms_boundaries_to_vertices(prisms):
    """
    Converts prisms boundaries to sets of vertices for each prism

    The vertices for each prism will be in the following order

    .. code-block::

                                7-------6
                               /|      /|
                              / |     / |
                             4-------5  |
        up   northing        |  |    |  |
        ^   ~                |  3----|--2
        |  /                 | /     | /
        | /                  |/      |/
         ------> easting     0-------1

    So the vertices of a single prism should be like:

    .. code-block:: python

        [w, s, bottom],
        [e, s, bottom],
        [e, n, bottom],
        [w, n, bottom],
        [w, s, top],
        [e, s, top],
        [e, n, top],
        [w, n, top],


    Parameters
    ----------
    prisms : 2d-array
        2d-array with the boundaries of a set of prisms. Each row of the array
        should contain the boundaries of a single prism in the following
        order: ``west``, ``east``, ``south``, ``north``, ``bottom``, ``top``.

    Returns
    -------
    vertices : 2d-array
        2D array with the coordinates of the vertices of the prism. Each row of
        the array corresponds to the coordinate of a single vertex in the
        following order: ``easting``, ``northing``, ``upward``.
        The shape of this array is ``(M, 3)``, where ``M`` is the total number
        of vertices in the whole set of prisms (number of prisms times 8).
        The order of the vertices is fixed to be compatible with VTK.

    Examples
    --------
    >>> _prisms_boundaries_to_vertices(np.array([[-1, 1, -2, 2, -3, 3]]))
    array([[-1., -2., -3.],
           [ 1., -2., -3.],
           [ 1.,  2., -3.],
           [-1.,  2., -3.],
           [-1., -2.,  3.],
           [ 1., -2.,  3.],
           [ 1.,  2.,  3.],
           [-1.,  2.,  3.]])
    >>> _prisms_boundaries_to_vertices(
    ...     np.array([[-1, 1, -2, 2, -3, 3], [-4, 4, -5, 5, -6, 6]])
    ... )
    array([[-1., -2., -3.],
           [ 1., -2., -3.],
           [ 1.,  2., -3.],
           [-1.,  2., -3.],
           [-1., -2.,  3.],
           [ 1., -2.,  3.],
           [ 1.,  2.,  3.],
           [-1.,  2.,  3.],
           [-4., -5., -6.],
           [ 4., -5., -6.],
           [ 4.,  5., -6.],
           [-4.,  5., -6.],
           [-4., -5.,  6.],
           [ 4., -5.,  6.],
           [ 4.,  5.,  6.],
           [-4.,  5.,  6.]])
    """
    # Get number of prisms
    n_prisms = prisms.shape[0]

    # Allocate vertices array
    vertices = np.empty((n_prisms, 8, 3))

    # Define a dictionary with the indices of the vertices that contain each
    # boundary of the prism.
    # For example, the west boundary is present only in the vertices
    # number 0, 3, 4 and 7.
    indices = {
        "west": (0, 3, 4, 7),
        "east": (1, 2, 5, 6),
        "south": (0, 1, 4, 5),
        "north": (2, 3, 6, 7),
        "bottom": (0, 1, 2, 3),
        "top": (4, 5, 6, 7),
    }

    # Assign the values to each vertex
    for i, boundary in enumerate(indices):
        # Determine at which component of the vertices should the current
        # boundary be assigned to.
        #   The west and east (i = 0 and i = 1) should go to the component 0.
        #   The south and north (i = 2 and i = 3) should go to the component 1.
        #   The bottom and top (i = 4 and i = 5) should go to the component 2.
        component = i // 2
        # Assign vertices components
        for vertex in indices[boundary]:
            vertices[:, vertex, component] = prisms[:, i]

    # Reshape the vertices array so it has (M, 3) shape, where M is the total
    # number of vertices.
    return vertices.reshape((n_prisms * 8, 3))
