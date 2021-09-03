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
import warnings
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd
import xarray as xr

from .. import prism_layer, prism_gravity


@pytest.fixture(params=("numpy", "xarray"))
def dummy_layer(request):
    """
    Generate dummy arrays for defining prism layers
    """
    easting = np.linspace(-1, 3, 5)
    northing = np.linspace(7, 13, 4)
    shape = (northing.size, easting.size)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(*shape)
    density = 2670 * np.ones(shape)
    if request.param == "xarray":
        easting = xr.DataArray(easting, dims=("easting",))
        northing = xr.DataArray(northing, dims=("northing",))
        reference, surface = xr.DataArray(reference), xr.DataArray(surface)
        density = xr.DataArray(density)
    return (easting, northing), surface, reference, density


@pytest.fixture
def prism_layer_with_holes(dummy_layer):  # pylint: disable=redefined-outer-name
    """
    Return a set of prisms with some missing elements

    The prisms are returned as a tuple of boundaries, ready to be passed to
    ``hm.prism_gravity``. They would represent the same prisms that the
    ``dummy_layer`` generated, but with two missing prisms: the ``(3, 3)`` and
    the ``(2, 1)``.
    """
    (easting, northing), surface, reference, density = dummy_layer
    layer = prism_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    indices = [(3, 3), (2, 1)]
    prisms = list(
        layer.prism_layer.get_prism((i, j))
        for i in range(4)
        for j in range(5)
        if (i, j) not in indices
    )
    density = list(
        density[i, j] for i in range(4) for j in range(5) if (i, j) not in indices
    )
    return prisms, density


def test_prism_layer(dummy_layer):  # pylint: disable=redefined-outer-name
    """
    Check if a layer of prisms is property constructed
    """
    (easting, northing), surface, reference, _ = dummy_layer
    layer = prism_layer((easting, northing), surface, reference)
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
    expected_bottom = np.zeros_like(surface)
    expected_top[1, 1], expected_bottom[1, 1] = reference, surface[1, 1]
    layer = prism_layer((easting, northing), surface, reference)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, easting)
    npt.assert_allclose(layer.northing, northing)
    npt.assert_allclose(layer.top, expected_top)
    npt.assert_allclose(layer.bottom, expected_bottom)


def test_prism_layer_invalid_surface_reference(
    dummy_layer,
):  # pylint: disable=redefined-outer-name
    """
    Check if invalid surface and/or reference are caught
    """
    coordinates, surface, reference, _ = dummy_layer
    # Surface with wrong shape
    surface_invalid = np.arange(20, dtype=float)
    with pytest.raises(ValueError):
        prism_layer(coordinates, surface_invalid, reference)
    # Reference with wrong shape
    reference_invalid = np.zeros(20)
    surface = np.arange(20, dtype=float).reshape(4, 5)
    with pytest.raises(ValueError):
        prism_layer(coordinates, surface, reference_invalid)


def test_prism_layer_properties(dummy_layer):  # pylint: disable=redefined-outer-name
    """
    Check passing physical properties to the prisms layer
    """
    coordinates, surface, reference, density = dummy_layer
    suceptibility = 0 * density + 1e-3
    layer = prism_layer(
        coordinates,
        surface,
        reference,
        properties={"density": density, "suceptibility": suceptibility},
    )
    npt.assert_allclose(layer.density, density)
    npt.assert_allclose(layer.suceptibility, suceptibility)


def test_prism_layer_no_regular_grid(
    dummy_layer,
):  # pylint: disable=redefined-outer-name
    """
    Check if error is raised if easting and northing are not regular
    """
    (easting, northing), surface, reference, _ = dummy_layer
    # Easting as non evenly spaced set of coordinates
    easting_invalid = easting.copy()
    easting_invalid[3] = -22
    with pytest.raises(ValueError):
        prism_layer(
            (easting_invalid, northing),
            surface,
            reference,
        )
    # Northing as non evenly spaced set of coordinates
    northing_invalid = northing.copy()
    northing_invalid[3] = -22
    northing[3] = 12.98
    with pytest.raises(ValueError):
        prism_layer(
            (easting, northing_invalid),
            surface,
            reference,
        )


def test_prism_layer_attributes():
    """
    Check attributes of the DatasetAccessorPrismsLayer class
    """
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20, dtype=float).reshape(4, 5)
    layer = prism_layer((easting, northing), surface, reference)
    assert layer.prism_layer.dims == ("northing", "easting")
    assert layer.prism_layer.spacing == (1, 0.5)
    assert layer.prism_layer.boundaries == (
        easting[0] - 0.25,
        easting[-1] + 0.25,
        northing[0] - 0.5,
        northing[-1] + 0.5,
    )
    assert layer.prism_layer.size == 20
    assert layer.prism_layer.shape == (4, 5)


def test_prism_layer_to_prisms():
    """
    Check the _to_prisms() method
    """
    coordinates = (np.array([0, 1]), np.array([0, 1]))
    reference = np.arange(4).reshape(2, 2)
    surface = (np.arange(4) + 10).reshape(2, 2)
    layer = prism_layer(coordinates, surface, reference)
    expected_prisms = [
        [-0.5, 0.5, -0.5, 0.5, 0, 10],
        [0.5, 1.5, -0.5, 0.5, 1, 11],
        [-0.5, 0.5, 0.5, 1.5, 2, 12],
        [0.5, 1.5, 0.5, 1.5, 3, 13],
    ]
    npt.assert_allclose(expected_prisms, layer.prism_layer._to_prisms())


def test_prism_layer_get_prism_by_index():
    """
    Check if the right prism is returned after index
    """
    coordinates = (np.array([0, 1]), np.array([0, 1]))
    reference = np.arange(4).reshape(2, 2)
    surface = (np.arange(4) + 10).reshape(2, 2)
    layer = prism_layer(coordinates, surface, reference)
    expected_prisms = [
        [[-0.5, 0.5, -0.5, 0.5, 0, 10], [0.5, 1.5, -0.5, 0.5, 1, 11]],
        [[-0.5, 0.5, 0.5, 1.5, 2, 12], [0.5, 1.5, 0.5, 1.5, 3, 13]],
    ]
    for i in range(2):
        for j in range(2):
            npt.assert_allclose(
                layer.prism_layer.get_prism((i, j)), expected_prisms[i][j]
            )


def test_nonans_prisms_mask(dummy_layer):  # pylint: disable=redefined-outer-name
    """
    Check if the mask for nonans prism is correctly created
    """
    (easting, northing), surface, reference, _ = dummy_layer
    shape = (northing.size, easting.size)

    # No nan in top nor bottom
    # ------------------------
    layer = prism_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    mask = layer.prism_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)

    # Nans in top only
    # ----------------
    layer = prism_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((1, 2), (2, 3)):
        layer.top[index] = np.nan
        expected_mask[index] = False
    mask = layer.prism_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)

    # Nans in bottom only
    # -------------------
    layer = prism_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((2, 1), (3, 2)):
        layer.bottom[index] = np.nan
        expected_mask[index] = False
    mask = layer.prism_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)

    # Nans in top and bottom
    # ----------------------
    layer = prism_layer((easting, northing), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((1, 2), (2, 3)):
        layer.top[index] = np.nan
        expected_mask[index] = False
    for index in ((1, 2), (2, 1), (3, 2)):
        layer.bottom[index] = np.nan
        expected_mask[index] = False
    mask = layer.prism_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)


def test_nonans_prisms_mask_property(
    dummy_layer,
):  # pylint: disable=redefined-outer-name
    """
    Check if the method masks the property and raises a warning
    """
    (easting, northing), surface, reference, density = dummy_layer
    shape = (northing.size, easting.size)

    # Nans in top and property (on the same prisms)
    # ---------------------------------------------
    expected_mask = np.ones_like(surface, dtype=bool)
    indices = ((1, 2), (2, 3))
    # Set some elements of surface and density as nans
    for index in indices:
        surface[index] = np.nan
        density[index] = np.nan
        expected_mask[index] = False
    layer = prism_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Check if no warning is raised
    with warnings.catch_warnings(record=True) as warn:
        mask = layer.prism_layer._get_nonans_mask(property_name="density")
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
    layer = prism_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    # Check if warning is raised
    with warnings.catch_warnings(record=True) as warn:
        mask = layer.prism_layer._get_nonans_mask(property_name="density")
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
    npt.assert_allclose(mask, expected_mask)


@pytest.mark.use_numba
@pytest.mark.parametrize("field", ["potential", "g_z"])
def test_prism_layer_gravity(
    field, dummy_layer
):  # pylint: disable=redefined-outer-name
    """
    Check if gravity method works as expected
    """
    coordinates = vd.grid_coordinates((1, 3, 7, 10), spacing=1, extra_coords=30.0)
    (easting, northing), surface, reference, density = dummy_layer
    layer = prism_layer(
        (easting, northing), surface, reference, properties={"density": density}
    )
    expected_result = prism_gravity(
        coordinates,
        prisms=layer.prism_layer._to_prisms(),
        density=density,
        field=field,
    )
    npt.assert_allclose(
        expected_result, layer.prism_layer.gravity(coordinates, field=field)
    )


@pytest.mark.use_numba
@pytest.mark.parametrize("field", ["potential", "g_z"])
def test_prism_layer_gravity_surface_nans(
    field, dummy_layer, prism_layer_with_holes
):  # pylint: disable=redefined-outer-name
    """
    Check if gravity method works as expected when surface has nans
    """
    coordinates = vd.grid_coordinates((1, 3, 7, 10), spacing=1, extra_coords=30.0)
    (easting, northing), surface, reference, density = dummy_layer
    # Create one layer that has nans on the surface array
    surface_w_nans = surface.copy()
    indices = [(3, 3), (2, 1)]
    for index in indices:
        surface_w_nans[index] = np.nan
    layer = prism_layer(
        (easting, northing), surface_w_nans, reference, properties={"density": density}
    )
    # Check if it generates the expected gravity field
    prisms, rho = prism_layer_with_holes
    npt.assert_allclose(
        layer.prism_layer.gravity(coordinates, field=field),
        prism_gravity(coordinates, prisms, rho, field=field),
    )


@pytest.mark.use_numba
@pytest.mark.parametrize("field", ["potential", "g_z"])
def test_prism_layer_gravity_density_nans(
    field, dummy_layer, prism_layer_with_holes
):  # pylint: disable=redefined-outer-name
    """
    Check if prisms is ignored after a nan is found in density array
    """
    coordinates = vd.grid_coordinates((1, 3, 7, 10), spacing=1, extra_coords=30.0)
    prisms_coords, surface, reference, density = dummy_layer
    # Create one layer that has nans on the density array
    indices = [(3, 3), (2, 1)]
    for index in indices:
        density[index] = np.nan
    layer = prism_layer(
        prisms_coords, surface, reference, properties={"density": density}
    )
    # Check if warning is raised after passing density with nans
    with warnings.catch_warnings(record=True) as warn:
        result = layer.prism_layer.gravity(coordinates, field=field)
        assert len(warn) == 1
    # Check if it generates the expected gravity field
    prisms, rho = prism_layer_with_holes
    npt.assert_allclose(
        result,
        prism_gravity(coordinates, prisms, rho, field=field),
    )
