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
    import vtk
except ImportError:
    vtk = None

try:
    import pyvista
except ImportError:
    pyvista = None


def prisms_to_pyvista(prisms, properties=None):
    """
    Create a ``pyvista.UnstructuredGrid`` from a set of prisms

    Parameters
    ----------
    prisms : list or 2d-array
        List or 2d-array with the boundaries of the prisms.
        Each row contains the boundaries of each prism in the following order:
        ``west``, ``east``, ``south``, ``north``, ``bottom``, ``top``.
    properties : dict or None (optional)
        Dictionary with the physical properties of the prisms.
        Each key should be a string and their value an array.
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

    >>> pv_grid = prisms_to_pyvista(
    ...     [0, 1e3, -2e3, 2e3, -5e3, -3e3],
    ...     properties={"density": 2670},
    ... )
    >>> pv_grid.n_cells
    1
    >>> pv_grid.n_points
    8
    >>> pv_grid.get_array("density")
    array([2670.])
    >>> pv_grid.cell_bounds(0)
    [0.0, 1000.0, -2000.0, 2000.0, -5000.0, -3000.0]


    """
    # Check if vkt and pyvista are installed
    msg = "Missing optional dependency '{}' required for generating pyvista grids."
    if vtk is None:
        raise ValueError(msg.format("vtk"))
    if pyvista is None:
        raise ValueError(msg.format("pyvista"))
    # Get prisms and number of prisms
    prisms = np.atleast_2d(prisms)
    n_prisms = prisms.shape[0]
    # Get the vertices of the prisms
    vertices = np.vstack([_get_prism_vertices(prism) for prism in prisms])
    # Get the cells for these prisms
    cells = _build_cells(n_prisms)
    # Define an array with the types of the cells
    cell_type = np.full(n_prisms, vtk.VTK_HEXAHEDRON)
    # Build the UnstructuredGrid
    pv_grid = pyvista.UnstructuredGrid(cells, cell_type, vertices)
    # Add properties to the grid
    if properties is not None:
        for name, prop in properties.items():
            # Convert the property to an array and ravel it
            prop = np.atleast_1d(prop).ravel()
            # Assign the property to the cell_data
            pv_grid.cell_data[name] = prop
    return pv_grid


def _build_cells(n_prisms):
    """
    Build the VTK cells for a given number of prisms

    We will represent each prism a single cell of ``VTK_HEXAHEDRON`` type.
    Each cell is an array with 9 elements:
      - the first one indicates the number of vertices that the cell will have
        (8 for prisms), and
      - the following 8 elements are the indices of the vertices that form the
        prism.

    The cells is a concatenation of every 9 elements cell array, resulting in
    a 1d-array of 9 times ``n_prisms`` elements.

    Parameters
    ----------
    n_prisms : int
        Number of prisms in the set

    Returns
    -------
    cells : 1d-array
        Array representing the cells for each one of the prisms.

    Examples
    --------

    >>> _build_cells(4)
    array([ 8,  0,  1,  2,  3,  4,  5,  6,  7,  8,  8,  9, 10, 11, 12, 13, 14,
           15,  8, 16, 17, 18, 19, 20, 21, 22, 23,  8, 24, 25, 26, 27, 28, 29,
           30, 31])
    """
    # Get total number of vertices
    n_vertices = n_prisms * 8
    # Build the indices array as a 2D array: each row contains the indices for
    # each prism
    indices = np.arange(n_vertices).reshape(n_prisms, 8)
    # Create a vertical array full of 8
    n_vertices_per_cell = np.full(fill_value=8, shape=(n_prisms, 1))
    # Stack the indices and n_vertices_per_cell arrays
    cells = np.hstack((n_vertices_per_cell, indices)).ravel()
    return cells


def _get_prism_vertices(prism):
    """
    Return the vertices of the given prism

    Parameters
    ----------
    prism : list or 1d-array
        List or 1d-array with the boundaries of a single prism in the following
        order: ``west``, ``east``, ``south``, ``north``, ``bottom``, ``top``.

    Returns
    -------
    vertices : 2d-array
        2D array with the coordinates of the vertices of the prism. Each row of
        the array corresponds to the coordinate of a single vertex in the
        following order: ``easting``, ``northing``, ``upward``.
        The order of the vertices is fixed to be compatible with VTK.

    Examples
    --------

    >>> _get_prism_vertices([-1, 1, -2, 2, -3, 3])
    array([[-1, -2, -3],
           [ 1, -2, -3],
           [ 1,  2, -3],
           [-1,  2, -3],
           [-1, -2,  3],
           [ 1, -2,  3],
           [ 1,  2,  3],
           [-1,  2,  3]])

    """
    w, e, s, n, bottom, top = prism[:]
    vertices = np.array(
        [
            [w, s, bottom],
            [e, s, bottom],
            [e, n, bottom],
            [w, n, bottom],
            [w, s, top],
            [e, s, top],
            [e, n, top],
            [w, n, top],
        ]
    )
    return vertices
