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
    vertices = np.vstack([_get_prism_vertices(prism) for prism in prisms])
    # Generate the cells for the hexahedrons
    cells = np.arange(n_prisms * 8).reshape([n_prisms, 8])
    # Build the UnstructuredGrid
    pv_grid = pyvista.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells}, vertices)
    # Add properties to the grid
    if properties is not None:
        for name, prop in properties.items():
            # Assign the property to the cell_data
            pv_grid.cell_data[name] = np.atleast_1d(prop).ravel()
    return pv_grid


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
