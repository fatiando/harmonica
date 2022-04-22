# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test functions from the visualization module.
"""
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from ..visualization.prism import prism_to_pyvista

try:
    import pyvista
except ImportError:
    pyvista = None


@pytest.mark.skipif(pyvista is not None, reason="pyvista must be missing")
def test_prism_to_pyvista_missing_pyvista():
    """
    Check error raise after calling prism_to_pyvista when pyvista is missing
    """
    prism = [0, 1, 0, 1, 0, 1]
    with pytest.raises(ImportError) as exception:
        prism_to_pyvista(prism)
    assert "'pyvista'" in str(exception.value)


@pytest.fixture(name="prisms")
def fixture_prisms():
    """
    Define a set of sample prisms
    """
    return [
        [1e3, 5e3, -2e3, 4e3, -10e3, -7e3],
        [200, 500, 600, 900, 100, 300],
        [-2e3, 0, 5e3, 15e3, -4e3, 2e3],
        [-10e3, -8e3, -10e3, -8e3, -10e3, -8e3],
    ]


@pytest.fixture(name="density", params=("numpy", "xarray"))
def fixture_density(request):
    """
    Return a density array for the sample prisms
    """
    density = [2670.0, 2900.0, 3300.0, 3200.0]
    if request.param == "xarray":
        density = xr.DataArray(density)
    return density


@pytest.fixture(name="susceptibility", params=("numpy", "xarray"))
def fixture_susceptibility(request):
    """
    Return a susceptibility array for the sample prisms
    """
    susceptibility = [2e-4, 2.5e-4, 5e-4, 1e-3]
    if request.param == "xarray":
        susceptibility = xr.DataArray(susceptibility)
    return susceptibility


@pytest.mark.skipif(pyvista is None, reason="requires pyvista")
def test_prism_to_pyvista(prisms, density):
    """
    Test output of prism_to_pyvista
    """
    pv_grid = prism_to_pyvista(prisms, properties={"density": density})
    assert pv_grid.n_cells == 4
    assert pv_grid.n_points == 32
    # Check coordinates of prisms
    for i, prism in enumerate(prisms):
        npt.assert_allclose(prism, pv_grid.cell_bounds(i))
    # Check properties of the prisms
    assert pv_grid.n_arrays == 1
    assert pv_grid.array_names == ["density"]
    npt.assert_allclose(pv_grid.get_array("density"), density)


@pytest.mark.skipif(pyvista is None, reason="requires pyvista")
@pytest.mark.parametrize("n_props", [0, 1, 2])
def test_prism_to_pyvista_properties(n_props, prisms, density, susceptibility):
    """
    Test prism_to_pyvista for different number of properties
    """
    properties = None
    if n_props == 1:
        properties = {"density": density}
    elif n_props == 2:
        properties = {"density": density, "susceptibility": susceptibility}
    pv_grid = prism_to_pyvista(prisms, properties=properties)
    # Check if the grid has the right properties
    if properties is None:
        assert pv_grid.n_arrays == 0
        assert pv_grid.array_names == []
    else:
        assert pv_grid.n_arrays == len(properties)
        assert set(pv_grid.array_names) == set(properties.keys())
        for prop in properties:
            npt.assert_allclose(pv_grid.get_array(prop), properties[prop])


@pytest.mark.skipif(pyvista is None, reason="requires pyvista")
def test_prism_to_pyvista_error_2d_property(prisms, density):
    """
    Test if prism_to_pyvista raises error on property as 2D array
    """
    density_2d = np.array(density).reshape((2, 2))
    with pytest.raises(ValueError):
        prism_to_pyvista(prisms, properties={"density": density_2d})


@patch("harmonica.visualization.prism.pyvista", None)
def test_prisms_pyvista_missing_error(prisms, density):
    """
    Check if prism_to_pyvista raises error if pyvista is missing
    """
    # Check if error is raised
    with pytest.raises(ImportError):
        prism_to_pyvista(prisms, properties={"density": density})
