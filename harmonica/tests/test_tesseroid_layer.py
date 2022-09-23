# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test tesseroids layer
"""
#
import warnings

import boule
import numpy as np
import numpy.testing as npt
import pytest
import verde as vd
import xarray as xr

from .. import tesseroid_gravity, tesseroid_layer


@pytest.fixture
def mean_earth_radius():
    """
    Return mean earth radius given by WGS84
    """
    return boule.WGS84.mean_radius


@pytest.fixture(params=("numpy", "reference-as-array", "xarray"))
def dummy_layer(mean_earth_radius, request):
    """
    Generate dummy array for defining tesseroid layers
    """
    latitude = np.linspace(-10, 10, 6)
    longitude = np.linspace(-10, 10, 5)
    shape = (latitude.size, longitude.size)
    surface = mean_earth_radius * np.ones(shape) + 1e3
    reference = mean_earth_radius
    if request.param == "reference-as-array":
        reference *= np.ones(shape)
    density = 2670 * np.ones(shape)
    if request.param == "xarray":
        latitude = xr.DataArray(latitude, dims=("latitude",))
        longitude = xr.DataArray(longitude, dims=("longitude",))
        reference, surface = xr.DataArray(reference), xr.DataArray(surface)
        density = xr.DataArray(density)
    return (longitude, latitude), surface, reference, density


@pytest.fixture
def tesseroid_layer_with_holes(dummy_layer):
    """
    Return a set of tesseroids with some missing elements

    The tesseroids are returned as a tuple of boundaries, ready to be passed to
    ``hm.tesseroid_gravity``.
    They would represent the same tesseroids that the ``dummy_layer``
    generated, but with two missing tesseroids: the ``(3, 3)`` and the
    ``(2, 1)``.
    """
    (longitude, latitude), surface, reference, density = dummy_layer
    layer = tesseroid_layer(
        (longitude, latitude), surface, reference, properties={"density": density}
    )
    indices = [(3, 3), (2, 1)]
    tesseroids = list(
        layer.tesseroid_layer.get_tesseroid((i, j))
        for i in range(6)
        for j in range(5)
        if (i, j) not in indices
    )
    density = list(
        density[i, j] for i in range(6) for j in range(5) if (i, j) not in indices
    )
    return tesseroids, density


@pytest.mark.parametrize(
    "longitude_w, longitude_e",
    [
        (0, 480),
        (0, 360),
        (-180, 180),
        (0, 360 - 18 / 2),
    ],
)
def test_tesseroid_overlap_wrong_coords(longitude_w, longitude_e, mean_earth_radius):
    """
    Check if the tesseroid are overlapped
    """
    latitude = np.linspace(-10, 10, 6)
    longitude = np.linspace(longitude_w, longitude_e, 21)
    shape = (latitude.size, longitude.size)
    surface = mean_earth_radius * np.ones(shape) + 1e3
    reference = mean_earth_radius * np.ones(shape)
    with pytest.raises(
        ValueError,
        match="Found invalid longitude coordinates that would create overlapping tesseroids around the globe.",
    ):
        tesseroid_layer((longitude, latitude), surface, reference)


@pytest.mark.parametrize(
    "longitude_w, longitude_e",
    [
        (0, 360 - 18),
        (-180, 180 - 18),
    ],
)
def test_tesseroid_overlap_right_coords(longitude_w, longitude_e, mean_earth_radius):
    """
    Check if the tesseroid are not overlapped
    """
    latitude = np.linspace(-10, 10, 6)
    longitude = np.linspace(longitude_w, longitude_e, 21)
    shape = (latitude.size, longitude.size)
    surface = mean_earth_radius * np.ones(shape) + 1e3
    reference = mean_earth_radius * np.ones(shape)
    tesseroid_layer((longitude, latitude), surface, reference)


def test_tesseroid_layer(dummy_layer, mean_earth_radius):
    """
    Check if the layer of tesseroids is property constructed
    """
    (longitude, latitude), surface, reference, _ = dummy_layer
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    assert "latitude" in layer.coords
    assert "longitude" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.latitude, latitude)
    npt.assert_allclose(layer.longitude, longitude)
    npt.assert_allclose(layer.top, surface)
    npt.assert_allclose(layer.bottom, reference)
    # Surface below reference on a single point
    surface[1, 1] = mean_earth_radius - 1e3  # reference is on mean_earth_radius
    expected_top = surface.copy()
    expected_bottom = mean_earth_radius * np.ones_like(surface)
    expected_top[1, 1], expected_bottom[1, 1] = mean_earth_radius, surface[1, 1]
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    assert "latitude" in layer.coords
    assert "longitude" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.latitude, latitude)
    npt.assert_allclose(layer.longitude, longitude)
    npt.assert_allclose(layer.top, expected_top)
    npt.assert_allclose(layer.bottom, expected_bottom)


def test_tesseroid_layer_invalid_surface_reference(dummy_layer):
    """
    Check if invalid surface and/or reference are caught
    """
    coordinates, surface, reference, _ = dummy_layer
    # Surface with wrong shape
    surface_invalid = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="Invalid surface array with shape"):
        tesseroid_layer(coordinates, surface_invalid, reference)
    # Reference with wrong shape
    reference_invalid = np.zeros(20)
    with pytest.raises(ValueError, match="Invalid reference array with shape"):
        tesseroid_layer(coordinates, surface, reference_invalid)


def test_tesseroid_leyer_properties(dummy_layer):
    """
    Check passing physical properties to the tesseroid layer
    """
    coordinates, surface, reference, density = dummy_layer
    suceptibility = 0 * density + 1e-3
    layer = tesseroid_layer(
        coordinates,
        surface,
        reference,
        properties={"density": density, "suceptibility": suceptibility},
    )
    npt.assert_allclose(layer.density, density)
    npt.assert_allclose(layer.suceptibility, suceptibility)


def test_tesseroid_layer_no_regular_grid(
    dummy_layer,
):
    """
    Check if error is raised if the latitude or longitude are not regular
    """

    (longitude, latitude), surface, reference, _ = dummy_layer
    # Longitude as not evenly spaced set of coordinates
    longitude_invalid = longitude.copy()
    longitude_invalid[3] = -22
    with pytest.raises(ValueError):
        tesseroid_layer(
            (longitude_invalid, latitude),
            surface,
            reference,
        )
    # Latitude as not evenly spaced set of coordinates
    latitude_invalid = latitude.copy()
    latitude_invalid[3] = -22
    with pytest.raises(ValueError):
        tesseroid_layer(
            (longitude, latitude_invalid),
            surface,
            reference,
        )


def test_tesseroi_layer_attibutes():
    """
    Check attributes of the DatasetAccessorTesseroidLayer class
    """
    latitude = np.linspace(-10, 10, 6)
    longitude = np.linspace(-10, 10, 5)
    shape = (latitude.size, longitude.size)
    ellipsoid = boule.WGS84
    surface = ellipsoid.mean_radius * np.ones(shape)
    reference = (surface - 1e3) * np.ones(shape)
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    assert layer.tesseroid_layer.dims == ("latitude", "longitude")
    assert layer.tesseroid_layer.spacing == (4, 5)
    assert layer.tesseroid_layer.boundaries == (
        longitude[0] - 2.5,
        longitude[-1] + 2.5,
        latitude[0] - 2,
        latitude[-1] + 2,
    )
    assert layer.tesseroid_layer.size == 30
    assert layer.tesseroid_layer.shape == (6, 5)


def test_tesseroid_layer_to_tesseroid():
    """
    Check the _to_tesseroid() method
    """
    latitude = np.linspace(-1, 1, 2)
    longitude = np.linspace(-2, 2, 2)
    shape = (latitude.size, longitude.size)
    ellipsoid = boule.WGS84
    surface = ellipsoid.mean_radius * np.ones(shape)
    reference = (surface - 1e3) * np.ones(shape)
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    expected_tesseroids = [
        [-4.0, 0.0, -2.0, 0.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
        [0.0, 4.0, -2.0, 0.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
        [-4.0, 0.0, 0.0, 2.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
        [0.0, 4.0, 0.0, 2.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
    ]
    npt.assert_allclose(expected_tesseroids, layer.tesseroid_layer._to_tesseroids())


def test_tesseroid_layer_get_tesseroid_by_index():
    """
    Check if the right tesseroid is returned after index
    """
    latitude = np.linspace(-1, 1, 2)
    longitude = np.linspace(-2, 2, 2)
    shape = (latitude.size, longitude.size)
    ellipsoid = boule.WGS84
    surface = ellipsoid.mean_radius * np.ones(shape)
    reference = (surface - 1e3) * np.ones(shape)
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    expected_tesseroids = [
        [
            [-4.0, 0.0, -2.0, 0.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
            [0.0, 4.0, -2.0, 0.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
        ],
        [
            [-4.0, 0.0, 0.0, 2.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
            [0.0, 4.0, 0.0, 2.0, ellipsoid.mean_radius - 1e3, ellipsoid.mean_radius],
        ],
    ]
    print(layer)
    for i in range(2):
        for j in range(2):
            print(i, j)
            print(
                layer.tesseroid_layer.get_tesseroid((i, j)), expected_tesseroids[i][j]
            )
            npt.assert_allclose(
                layer.tesseroid_layer.get_tesseroid((i, j)), expected_tesseroids[i][j]
            )


def test_nonans_tesseroid_mask(dummy_layer):
    """
    Check if the mask for nonans tesseroid is correctly created
    """
    (longitude, latitude), surface, reference, _ = dummy_layer
    shape = (latitude.size, longitude.size)
    # No nan in top or bottom
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    mask = layer.tesseroid_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)
    # Nans in the top only
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((2, 1), (3, 2)):
        layer.top[index] = np.nan
        expected_mask[index] = False
    mask = layer.tesseroid_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)
    # Nans in the bottom only
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((2, 1), (3, 2)):
        layer.bottom[index] = np.nan
        expected_mask[index] = False
    mask = layer.tesseroid_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)
    # Nans in the top and bottom
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    expected_mask = np.ones(shape, dtype=bool)
    for index in ((1, 2), (2, 3)):
        layer.top[index] = np.nan
        expected_mask[index] = False
    for index in ((1, 2), (2, 1), (3, 2)):
        layer.bottom[index] = np.nan
        expected_mask[index] = False
    mask = layer.tesseroid_layer._get_nonans_mask()
    npt.assert_allclose(mask, expected_mask)


def test_nonans_tesseroid_mask_property(
    dummy_layer,
):
    """
    Check if the method masks the property and raises a warning
    """
    coordinates, surface, reference, density = dummy_layer
    shape = (coordinates[1].size, coordinates[0].size)
    # Nans in top and property (on the same tesseroid)
    expected_mask = np.ones_like(surface, dtype=bool)
    indices = ((1, 2), (2, 3))
    # Set some elements of surface and density as nans
    for index in indices:
        surface[index] = np.nan
        density[index] = np.nan
        expected_mask[index] = False
    layer = tesseroid_layer(
        coordinates, surface, reference, properties={"density": density}
    )
    # Check if no warning is raised
    with warnings.catch_warnings(record=True) as warn:
        mask = layer.tesseroid_layer._get_nonans_mask(property_name="density")
        assert len(warn) == 0
    npt.assert_allclose(mask, expected_mask)
    # Nans in top and property (not precisely on the same tesseroid)
    surface = np.arange(30, dtype=float).reshape(shape)
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
    layer = tesseroid_layer(
        coordinates, surface, reference, properties={"density": density}
    )
    # Check if warning is raised
    with warnings.catch_warnings(record=True) as warn:
        mask = layer.tesseroid_layer._get_nonans_mask(property_name="density")
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
    npt.assert_allclose(mask, expected_mask)


@pytest.mark.use_numba
@pytest.mark.parametrize("field", ["potential", "g_z"])
def test_tesseroid_layer_gravity(field, dummy_layer):
    """
    Check if gravity method works as expected
    """
    (longitude, latitude), surface, reference, density = dummy_layer
    # Create a regular grid of computation points located at 10km above surface
    grid_coords = vd.grid_coordinates(
        (-10, 10, -10, 10), spacing=7, extra_coords=(surface[0] + 10e3)
    )
    layer = tesseroid_layer(
        (longitude, latitude), surface, reference, properties={"density": density}
    )
    expected_result = tesseroid_gravity(
        grid_coords,
        tesseroids=layer.tesseroid_layer._to_tesseroids(),
        density=density,
        field=field,
    )
    npt.assert_allclose(
        expected_result, layer.tesseroid_layer.gravity(grid_coords, field=field)
    )


@pytest.mark.use_numba
@pytest.mark.parametrize("field", ["potential", "g_z"])
def test_tesseroid_layer_gravity_surface_nans(
    field, dummy_layer, tesseroid_layer_with_holes
):
    """
    Check if gravity method works as expected when surface has nans
    """
    (longitude, latitude), surface, reference, density = dummy_layer
    grid_coords = vd.grid_coordinates(
        (-10, 10, -10, 10), spacing=7, extra_coords=(surface[0] + 10e3)
    )
    # Create one layer that has nans on the surface array
    surface_w_nans = surface.copy()
    indices = [(3, 3), (2, 1)]
    for index in indices:
        surface_w_nans[index] = np.nan
    layer = tesseroid_layer(
        (longitude, latitude),
        surface_w_nans,
        reference,
        properties={"density": density},
    )
    # Check if it generates the expected gravity field
    tesseroids, rho = tesseroid_layer_with_holes
    npt.assert_allclose(
        layer.tesseroid_layer.gravity(grid_coords, field=field),
        tesseroid_gravity(grid_coords, tesseroids, rho, field=field),
    )


@pytest.mark.use_numba
@pytest.mark.parametrize("field", ["potential", "g_z"])
def test_tesseroid_layer_gravity_density_nans(
    field, dummy_layer, tesseroid_layer_with_holes
):
    """
    Check if tesseroid is ignored after a nan is found in density array
    """
    (longitude, latitude), surface, reference, density = dummy_layer
    grid_coords = vd.grid_coordinates(
        (-10, 10, -10, 10), spacing=7, extra_coords=(surface[0] + 10e3)
    )
    # Create one layer that has nans on the density array
    indices = [(3, 3), (2, 1)]
    for index in indices:
        density[index] = np.nan
    layer = tesseroid_layer(
        (longitude, latitude), surface, reference, properties={"density": density}
    )
    # Check if warning is raised after passing density with nans
    with warnings.catch_warnings(record=True) as warn:
        result = layer.tesseroid_layer.gravity(grid_coords, field=field)
        assert len(warn) == 1
    # Check if it generates the expected gravity field
    tesseroids, rho = tesseroid_layer_with_holes
    npt.assert_allclose(
        result,
        tesseroid_gravity(grid_coords, tesseroids, rho, field=field),
    )
