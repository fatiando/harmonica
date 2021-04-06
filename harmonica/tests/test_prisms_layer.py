# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# pylint: disable=protected-access
"""
Test prisms layer
"""
import pytest
import warnings
import numpy as np
import numpy.testing as npt
import verde as vd

from .. import prisms_layer, prism_gravity


def test_prisms_layer():
    """
    Check if a layer of prisms is property constructed
    """
    easting = np.linspace(-1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(4, 5)
    layer = prisms_layer((easting, northing), surface, reference)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, easting)
    npt.assert_allclose(layer.northing, northing)
    npt.assert_allclose(layer.top, surface)
    npt.assert_allclose(layer.bottom, reference)
    # Surface below reference on a single point
    surface[1, 1] = -1
    expected_top = surface.copy()
    expected_bottom = reference * np.ones_like(surface)
    expected_top[1, 1], expected_bottom[1, 1] = reference, surface[1, 1]
    layer = prisms_layer((easting, northing), surface, reference)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, easting)
    npt.assert_allclose(layer.northing, northing)
    npt.assert_allclose(layer.top, expected_top)
    npt.assert_allclose(layer.bottom, expected_bottom)


def test_prisms_layer_invalid_surface_reference():
    """
    Check if invalid surface and/or reference are caught
    """
    coordinates = np.linspace(-1, 3, 5), np.linspace(7, 10, 4)
    # Surface with wrong shape
    reference = 0
    surface = np.arange(20, dtype=float)
    with pytest.raises(ValueError):
        prisms_layer(coordinates, surface, reference)
    # Reference with wrong shape
    reference = np.zeros(20)
    surface = np.arange(20, dtype=float).reshape(4, 5)
    with pytest.raises(ValueError):
        prisms_layer(coordinates, surface, reference)


def test_prisms_layer_properties():
    """
    Check passing physical properties to the prisms layer
    """
    easting = np.linspace(-1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(4, 5)
    density = 2670 * np.ones_like(surface)
    suceptibility = np.arange(20, dtype=float).reshape(4, 5)
    layer = prisms_layer(
        (easting, northing),
        surface,
        reference,
        properties={"density": density, "suceptibility": suceptibility},
    )
    npt.assert_allclose(layer.density, density)
    npt.assert_allclose(layer.suceptibility, suceptibility)


def test_prisms_layer_attributes():
    """
    Check attributes of the DatasetAccessorPrismsLayer class
    """
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(4, 5)
    layer = prisms_layer((easting, northing), surface, reference)
    assert layer.prisms_layer.dims == ("northing", "easting")
    assert layer.prisms_layer.spacing == (1, 0.5)
    assert layer.prisms_layer.boundaries == (
        easting[0] - 0.25,
        easting[-1] + 0.25,
        northing[0] - 0.5,
        northing[-1] + 0.5,
    )
    assert layer.prisms_layer.size == 20
    assert layer.prisms_layer.shape == (4, 5)


def test_prisms_layer_to_prisms():
    """
    Check the _to_prisms() method
    """
    coordinates = (np.array([0, 1]), np.array([0, 1]))
    reference = np.arange(4).reshape(2, 2)
    surface = (np.arange(4) + 10).reshape(2, 2)
    layer = prisms_layer(coordinates, surface, reference)
    expected_prisms = [
        [-0.5, 0.5, -0.5, 0.5, 0, 10],
        [0.5, 1.5, -0.5, 0.5, 1, 11],
        [-0.5, 0.5, 0.5, 1.5, 2, 12],
        [0.5, 1.5, 0.5, 1.5, 3, 13],
    ]
    npt.assert_allclose(expected_prisms, layer.prisms_layer._to_prisms())


def test_prisms_layer_get_prism_by_index():
    """
    Check if the right prism is returned after index
    """
    coordinates = (np.array([0, 1]), np.array([0, 1]))
    reference = np.arange(4).reshape(2, 2)
    surface = (np.arange(4) + 10).reshape(2, 2)
    layer = prisms_layer(coordinates, surface, reference)
    expected_prisms = [
        [[-0.5, 0.5, -0.5, 0.5, 0, 10], [0.5, 1.5, -0.5, 0.5, 1, 11]],
        [[-0.5, 0.5, 0.5, 1.5, 2, 12], [0.5, 1.5, 0.5, 1.5, 3, 13]],
    ]
    for i in range(2):
        for j in range(2):
            npt.assert_allclose(
                layer.prisms_layer.get_prism((i, j)), expected_prisms[i][j]
            )


def test_nonans_prisms_mask():
    """
    Check if the mask for nonans prism is correctly created
    """
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    shape = (northing.size, easting.size)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(shape)

    # No nan in top nor bottom
    # ------------------------
    layer = prisms_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    mask = layer.prisms_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)

    # Nans in top only
    # ----------------
    layer = prisms_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((1, 2), (2, 3)):
        layer.top[index] = np.nan
        expected_mask[index] = False
    mask = layer.prisms_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)

    # Nans in bottom only
    # -------------------
    layer = prisms_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((2, 1), (3, 2)):
        layer.bottom[index] = np.nan
        expected_mask[index] = False
    mask = layer.prisms_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)

    # Nans in top and bottom
    # ----------------------
    layer = prisms_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((1, 2), (2, 3)):
        layer.top[index] = np.nan
        expected_mask[index] = False
    for index in ((1, 2), (2, 1), (3, 2)):
        layer.bottom[index] = np.nan
        expected_mask[index] = False
    mask = layer.prisms_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)


def test_nonans_prisms_mask_property():
    """
    Check if the method masks the property and raises a warning
    """
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    shape = (northing.size, easting.size)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(shape)
    density = 2670 * np.ones_like(surface)

    # Nans in top and property (on the same prisms)
    # ---------------------------------------------
    expected_mask = np.ones_like(surface, dtype=bool)
    indices = ((1, 2), (2, 3))
    # Set some elements of surface and density as nans
    for index in indices:
        surface[index] = np.nan
        density[index] = np.nan
        expected_mask[index] = False
    layer = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Check if no warning is raised
    with warnings.catch_warnings(record=True) as warn:
        mask = layer.prisms_layer._get_nonans_mask(property_name="density")
        assert len(warn) == 0
    npt.assert_allclose(mask, expected_mask)

    # Nans in top and property (not precisely on the same prisms)
    # -----------------------------------------------------------
    surface = np.arange(20, dtype=float).reshape(shape)
    density = 2670 * np.ones_like(surface)
    expected_mask = np.ones_like(surface, dtype=bool)
    # Set some elements of surface as nans
    indices = ((1, 2), (2, 3))
    for index in indices:
        surface[index] = np.nan
        expected_mask[index] = False
    # Set a different set of elements of density as nans
    indices = ((2, 2), (0, 1))
    for index in indices:
        density[index] = np.nan
        expected_mask[index] = False
    layer = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Check if warning is raised
    with warnings.catch_warnings(record=True) as warn:
        mask = layer.prisms_layer._get_nonans_mask(property_name="density")
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
    npt.assert_allclose(mask, expected_mask)


@pytest.mark.use_numba
def test_prisms_layer_gravity():
    """
    Check if gravity method works as expected
    """
    coordinates = vd.grid_coordinates((1, 3, 7, 10), spacing=1, extra_coords=30.0)
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    shape = (northing.size, easting.size)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(shape)
    density = np.ones_like(surface, dtype=float)
    layer = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    for field in ("potential", "g_z"):
        expected_result = prism_gravity(
            coordinates,
            prisms=layer.prisms_layer._to_prisms(),
            density=density,
            field=field,
        )
        npt.assert_allclose(
            expected_result, layer.prisms_layer.gravity(coordinates, field=field)
        )


@pytest.mark.use_numba
def test_prisms_layer_gravity_with_nans():
    """
    Check if gravity method works as expected when one of the prisms has nans
    """
    coordinates = vd.grid_coordinates((1, 3, 7, 10), spacing=1, extra_coords=30.0)
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    shape = (northing.size, easting.size)
    reference = 0
    # Create one layer that has nans on the surface array
    surface = np.arange(20, dtype=float).reshape(shape)
    indices = [(3, 3), (2, 1)]
    for index in indices:
        surface[index] = np.nan
    density = np.ones_like(surface, dtype=float)
    layer_nans = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Create one layer that has zero density but no nans
    surface = np.arange(20, dtype=float).reshape(shape)
    density = np.ones_like(surface, dtype=float)
    for index in indices:
        density[index] = 0
    layer_nonans = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Check if the two layers generate the same gravity field
    for field in ("potential", "g_z"):
        npt.assert_allclose(
            layer_nans.prisms_layer.gravity(coordinates, field=field),
            layer_nonans.prisms_layer.gravity(coordinates, field=field),
        )


def test_prisms_layer_gravity_density_nans():
    """
    Check if prisms is ignored after a nan is found in density array
    """
    coordinates = vd.grid_coordinates((1, 3, 7, 10), spacing=1, extra_coords=30.0)
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    shape = (northing.size, easting.size)
    reference = 0
    # Create one layer that has nans on the density array
    surface = np.arange(20, dtype=float).reshape(shape)
    indices = [(3, 3), (2, 1)]
    for index in indices:
        surface[index] = np.nan
    density = np.ones_like(surface, dtype=float)
    layer_nans = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Create one layer that has zero density but no nans
    surface = np.arange(20, dtype=float).reshape(shape)
    density = np.ones_like(surface, dtype=float)
    for index in indices:
        density[index] = 0
    layer_nonans = prisms_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Check if the two layers generate the same gravity field
    for field in ("potential", "g_z"):
        npt.assert_allclose(
            layer_nans.prisms_layer.gravity(coordinates, field=field),
            layer_nonans.prisms_layer.gravity(coordinates, field=field),
        )
