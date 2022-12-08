# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Testing isostasy calculation
"""
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from ..isostasy import isostatic_moho_airy


@pytest.mark.parametrize("reference_depth", (0, 30e3))
def test_airy_without_load(reference_depth):
    """
    Root should be zero for no equivalent topography (zero basement, no layers)
    """
    basement = np.zeros(20, dtype=np.float64)
    npt.assert_equal(
        isostatic_moho_airy(basement, reference_depth=reference_depth), reference_depth
    )


@pytest.mark.parametrize("shape", (20, (20, 31), (20, 31, 2)))
def test_airy_array_shape_preserved(shape):
    """
    Check that the shape of the topography is preserved
    """
    basement = np.zeros(shape, dtype=np.float64)
    assert isostatic_moho_airy(basement).shape == basement.shape


@pytest.fixture(name="basement", params=("numpy", "xarray"))
def fixture_basement(request):
    """
    Return a basement array
    """
    basement = np.array([-2, -1, 0, 1, 2, 3], dtype=float)
    if request.param == "xarray":
        basement = xr.DataArray(basement)
    return basement


@pytest.fixture(name="water", params=("numpy", "xarray"))
def fixture_water(request):
    """
    Return thickness and density for a water layer
    """
    thickness = np.array([2, 1, 0, 0, 0, 0], dtype=float)
    density = 0.5
    if request.param == "xarray":
        thickness = xr.DataArray(thickness)
    return thickness, density


@pytest.fixture(name="water_array", params=("xarray", "xarray"))
def fixture_water_array(request):
    """
    Return thickness and density for a water layer as array
    """
    thickness = np.array([2, 1, 0, 0, 0, 0], dtype=float)
    density = np.array([0.5, 0.4, 0.5, 0.4, 0.5, 0.5], dtype=float)
    if request.param == "xarray":
        thickness = xr.DataArray(thickness)
        density = xr.DataArray(density)
    return thickness, density


@pytest.fixture(name="sediments", params=("numpy", "xarray"))
def fixture_sediments(request):
    """
    Return thickness and density for a sediments layer
    """
    thickness = np.array([1, 2, 1, 0, 1.5, 0], dtype=float)
    density = 0.75
    if request.param == "xarray":
        thickness = xr.DataArray(thickness)
    return thickness, density


@pytest.fixture(name="sediments_array", params=("xarray", "xarray"))
def fixture_sediments_array(request):
    """
    Return thickness and density for a sediments layer
    """
    thickness = np.array([1, 2, 1, 0, 1.5, 0], dtype=float)
    density = np.array([0.76, 0.75, 0.76, 0.75, 0.76, 0.75], dtype=float)
    if request.param == "xarray":
        thickness = xr.DataArray(thickness)
        density = xr.DataArray(density)
    return thickness, density


def test_airy_no_layer(basement):
    "Use no layer to check the calculations"
    layers = None
    root = isostatic_moho_airy(
        basement,
        layers=layers,
        density_crust=1,
        density_mantle=3,
        reference_depth=10,
    )
    true_root = np.array([9, 9.5, 10, 10.5, 11, 11.5])
    npt.assert_equal(root, true_root)


def test_airy_single_layer(basement, water):
    "Use a simple basement + water model to check the calculations"
    thickness_water, density_water = water
    layers = {"water": (thickness_water, density_water)}
    root = isostatic_moho_airy(
        basement,
        layers=layers,
        density_crust=1,
        density_mantle=3,
        reference_depth=0,
    )
    true_root = np.array([-0.5, -0.25, 0, 0.5, 1, 1.5])
    npt.assert_equal(root, true_root)
    if isinstance(root, xr.DataArray):
        assert root.attrs["density_water"] == density_water


def test_airy_single_layer_array(basement, water_array):
    """
    Use a simple basement + water model with density as array to check
    the calculations
    """
    thickness_water, density_water = water_array
    layers = {"water": (thickness_water, density_water)}
    root = isostatic_moho_airy(
        basement,
        layers=layers,
        density_crust=np.array([1, 2, 1, 1, 2, 1], dtype=float),
        density_mantle=np.array([3, 3, 3, 3, 3, 3], dtype=float),
        reference_depth=0,
    )
    true_root = np.array([-0.5, -1.6, 0.0, 0.5, 4.0, 1.5])
    npt.assert_allclose(root, true_root, rtol=1e-10, atol=0)
    if isinstance(root, xr.DataArray):
        assert (root.attrs["density_water"] == density_water).all()


def test_airy_multiple_layers(basement, water, sediments):
    "Check isostasy function against a model with multiple layers"
    thickness_water, density_water = water
    thickness_sediments, density_sediments = sediments
    layers = {
        "water": (thickness_water, density_water),
        "sediments": (thickness_sediments, density_sediments),
    }
    root = isostatic_moho_airy(
        basement,
        layers=layers,
        density_crust=1,
        density_mantle=3,
        reference_depth=0,
    )
    true_root = np.array([-0.125, 0.5, 0.375, 0.5, 1.5625, 1.5])
    npt.assert_equal(root, true_root)
    if isinstance(root, xr.DataArray):
        assert root.attrs["density_water"] == density_water
        assert root.attrs["density_sediments"] == density_sediments


def test_airy_multiple_layers_array(basement, water_array, sediments_array):
    """
    Check isostasy function against a model with multiple layers with density
    as array
    """
    thickness_water, density_water = water_array
    thickness_sediments, density_sediments = sediments_array
    layers = {
        "water": (thickness_water, density_water),
        "sediments": (thickness_sediments, density_sediments),
    }
    root = isostatic_moho_airy(
        basement,
        layers=layers,
        density_crust=np.array([1, 2, 1, 1, 2, 1], dtype=float),
        density_mantle=np.array([3, 3, 3, 3, 3, 3], dtype=float),
        reference_depth=0,
    )
    true_root = np.array([-0.12, -0.1, 0.38, 0.5, 5.14, 1.5])
    npt.assert_allclose(root, true_root, rtol=1e-10, atol=0)
    if isinstance(root, xr.DataArray):
        assert (root.attrs["density_water"] == density_water).all()
        assert (root.attrs["density_sediments"] == density_sediments).all()
