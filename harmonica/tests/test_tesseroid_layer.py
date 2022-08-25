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
import xarray as xr

from .. import tesseroid_gravity, tesseroid_layer


@pytest.fixture(params=("numpy", "xarray"))
def dummy_layer(request):
    """
    Generate dummy array for defining tesseroid layers
    """
    latitude = np.linspace(-10, 10, 5)
    longitude = np.linspace(-10, 10, 5)
    shape = (latitude.size, longitude.size)
    ellipsoid = boule.WGS84
    surface = ellipsoid.mean_radius * np.ones(shape)
    reference = (surface - 1e3) * np.ones(shape)
    density = 2670 * np.ones(shape)
    if request.param == "xarray":
        latitude = xr.DataArray(latitude, dims=("latitude",))
        longitude = xr.DataArray(longitude, dims=("longitude",))
        reference, surface = xr.DataArray(reference), xr.DataArray(surface)
        density = xr.DataArray(density)
    return (latitude, longitude), surface, reference, density


def test_tesseroid_layer(dummy_layer):
    """
    Check if the layer of tesseroids is property constructed
    """
    (latitude, longitude), surface, reference, _ = dummy_layer
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
    surface[1, 1] = reference[1, 1] - 1e3
    expected_top = surface.copy()
    expected_bottom = reference.copy()
    expected_top[1, 1], expected_bottom[1, 1] = reference[1, 1], surface[1, 1]
    layer = tesseroid_layer((longitude, latitude), surface, reference)
    assert "latitude" in layer.coords
    assert "longitude" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.latitude, latitude)
    npt.assert_allclose(layer.longitude, longitude)
    npt.assert_allclose(layer.top, expected_top)
    npt.assert_allclose(layer.bottom, expected_bottom)
