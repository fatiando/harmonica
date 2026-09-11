# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Unit tests for padding functions
"""

import re

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

from harmonica.filters import fft, ifft, pad, unpad
from harmonica.filters._padding import _pad_coordinates, _pad_width_to_slice


@pytest.fixture
def sample_da_2d():
    """
    Defines a 2D sample xarray.DataArray
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.arange(11 * 17, dtype=float).reshape(17, 11)
    # Create one xr.DataArray for each coordinate and add spacing and
    # direct_lag attributes to them
    dx, dy = x[1] - x[0], y[1] - y[0]
    x = xr.DataArray(
        x, coords={"x": x}, dims=("x",), attrs={"direct_lag": 3.0, "spacing": dx}
    )
    y = xr.DataArray(
        y, coords={"y": y}, dims=("y",), attrs={"direct_lag": -2.1, "spacing": dy}
    )
    return xr.DataArray(z, coords={"x": x, "y": y}, dims=("y", "x"))


def test_pad_coordinates(sample_da_2d):
    """
    Test pad_coordinates function
    """
    coords = sample_da_2d.coords
    # Pad a single coordinate
    padded_coords = _pad_coordinates(coords, {"x": 3})
    npt.assert_allclose(padded_coords["x"], np.linspace(-3, 13, 17))
    npt.assert_allclose(padded_coords["y"], coords["y"])
    # Pad two coordinates
    padded_coords = _pad_coordinates(coords, {"x": 2, "y": 3})
    npt.assert_allclose(padded_coords["x"], np.linspace(-2, 12, 15))
    npt.assert_allclose(padded_coords["y"], np.linspace(-5.5, 5.5, 23))
    # Pad a single coordinate asymmetrically
    padded_coords = _pad_coordinates(coords, {"x": (3, 2)})
    npt.assert_allclose(padded_coords["x"], np.linspace(-3, 12, 16))
    npt.assert_allclose(padded_coords["y"], coords["y"])
    # Pad two coordinates asymmetrically
    padded_coords = _pad_coordinates(coords, {"x": (2, 1), "y": (3, 4)})
    npt.assert_allclose(padded_coords["x"], np.linspace(-2, 11, 14))
    npt.assert_allclose(padded_coords["y"], np.linspace(-5.5, 6, 24))


def test_pad_coordinates_invalid(sample_da_2d):
    """
    Test if pad_coordinates raises error after unevenly spaced coords
    """
    x = sample_da_2d.coords["x"].values.copy()
    x[3] += 0.1
    sample_da_2d = sample_da_2d.assign_coords({"x": x})
    msg = re.escape("Invalid 'x' coordinates: they must be evenly spaced.")
    with pytest.raises(ValueError, match=msg):
        _pad_coordinates(sample_da_2d.coords, pad_width={"x": 2})


def test_pad_with_kwargs(sample_da_2d):
    """
    Test pad function by passing pad_width as kwargs
    """
    padded_da = pad(sample_da_2d, x=2, y=1)
    assert padded_da.shape == (19, 15)
    npt.assert_allclose(padded_da.values[:1, :], 0)
    npt.assert_allclose(padded_da.values[-1:, :], 0)
    npt.assert_allclose(padded_da.values[:, :2], 0)
    npt.assert_allclose(padded_da.values[:, -2:], 0)
    npt.assert_allclose(padded_da.values[1:-1, 2:-2], sample_da_2d)
    npt.assert_allclose(padded_da.x, np.linspace(-2, 12, 15))
    npt.assert_allclose(padded_da.y, np.linspace(-4.5, 4.5, 19))


def test_pad_with_pad_width(sample_da_2d):
    """
    Test pad function by passing pad_width as argument
    """
    pad_width = {"x": (2, 3), "y": (1, 3)}
    padded_da = pad(sample_da_2d, pad_width)
    assert padded_da.shape == (21, 16)
    npt.assert_allclose(padded_da.values[:1, :], 0)
    npt.assert_allclose(padded_da.values[-3:, :], 0)
    npt.assert_allclose(padded_da.values[:, :2], 0)
    npt.assert_allclose(padded_da.values[:, -3:], 0)
    npt.assert_allclose(padded_da.values[1:-3, 2:-3], sample_da_2d)
    npt.assert_allclose(padded_da.x, np.linspace(-2, 13, 16))
    npt.assert_allclose(padded_da.y, np.linspace(-4.5, 5.5, 21))


@pytest.mark.parametrize(
    "pad_width",
    [
        {"x": 2, "y": 3},
        {"x": 2},
        {"y": 3},
        {"x": (2, 3), "y": 3},
        {"x": (2, 3), "y": (1, 3)},
        {"x": (2, 3)},
        {"y": (1, 3)},
    ],
)
def test_coordinates_attrs_after_pad(sample_da_2d, pad_width):
    """
    Test if the attributes of the coordinates are preserved after padding
    and if the pad_width has been added
    """
    padded_da = pad(sample_da_2d, pad_width)
    # Check if the attrs in sample_da_2d is a subset of the attrs in padded_da
    assert sample_da_2d.x.attrs.items() <= padded_da.x.attrs.items()
    assert sample_da_2d.y.attrs.items() <= padded_da.y.attrs.items()
    # Check if pad_width has been added to the attrs of each coordinate
    for coord, width in pad_width.items():
        assert padded_da.coords[coord].attrs["pad_width"] == width


@pytest.mark.parametrize(
    "pad_width",
    [
        {"x": 2, "y": 3},
        {"x": 2},
        {"y": 3},
        {"x": (2, 3), "y": 3},
        {"x": (2, 3), "y": (1, 3)},
        {"x": (2, 3)},
        {"y": (1, 3)},
    ],
)
def test_pad_unpad_round_trip(sample_da_2d, pad_width):
    """
    Test if applying pad and then unpad returns the original array
    """
    unpadded = unpad(pad(sample_da_2d, pad_width))
    xrt.assert_allclose(sample_da_2d, unpadded)


def test_unpad_invalid_array(sample_da_2d):
    """
    Test if error is raised when a not padded array is passed to unpad
    """
    msg = (
        "The passed array doesn't seem to be a padded one: the 'pad_width' "
        + "attribute was missing on every one of its coordinates. "
    )
    with pytest.raises(ValueError, match=msg):
        unpad(sample_da_2d)


@pytest.mark.parametrize(
    ("pad_width", "size", "expected_slice"),
    [
        ((1, 1), 4, slice(1, 3)),
        ((1, 2), 5, slice(1, 3)),
        ((2, 3), 10, slice(2, 7)),
        (2, 10, slice(2, 8)),
    ],
)
def test_pad_width_to_slice(pad_width, size, expected_slice):
    """
    Test if _pad_width_to_slice work as expected
    """
    assert _pad_width_to_slice(pad_width, size) == expected_slice


@pytest.mark.parametrize(
    "kwargs",
    [{"x": 1, "y": 1}, {"pad_width": {"x": 1, "y": 1}}],
    ids=["pad_width_as_kwargs", "pad_width_as_argument"],
)
def test_unpad_custom_path_width(sample_da_2d, kwargs):
    """
    Test the behaviour of unpad when passing a custom pad_width
    """
    unpadded = unpad(sample_da_2d, **kwargs)
    assert unpadded.shape == (15, 9)
    npt.assert_allclose(unpadded.x, np.linspace(1, 9, 9))
    npt.assert_allclose(unpadded.y, np.linspace(-3.5, 3.5, 15))


@pytest.mark.parametrize(
    "pad_width_arg",
    [None, "argument", "kwargs"],
    ids=["pad_width_none", "pad_width_as_arg", "pad_width_as_kwargs"],
)
def test_unpad_pop_pad_width_attributes(sample_da_2d, pad_width_arg):
    """
    Check if the unpadded array has no pad_width attributes
    """
    pad_width = {"x": 2, "y": 1}
    padded = pad(sample_da_2d, pad_width)
    if pad_width_arg is None:
        unpadded = unpad(padded)
    elif pad_width_arg == "argument":
        unpadded = unpad(padded, pad_width=pad_width)
    else:
        unpadded = unpad(padded, **pad_width)
    # Check if unpadded doesn't have the pad_width attributes
    for dim in unpadded.coords:
        assert "pad_width" not in unpadded.coords[dim].attrs


@pytest.mark.parametrize(
    "pad_width",
    [
        {"x": 4, "y": 3},
        {"x": 4},
        {"x": 0},
        {"y": 3},
        {"x": 4, "y": 0},
        {"x": (4, 3), "y": 3},
        {"x": (4, 3), "y": (5, 3)},
        {"x": (4, 3)},
        {"y": (5, 3)},
        {"x": (0, 3), "y": (5, 0)},
    ],
)
def test_unpad_ifft_fft_pad_round_trip(sample_da_2d, pad_width):
    """
    Test if the round trip with padding and unpadding works

    This test passes a custom ``pad_width`` to the ``unpad`` function because
    the ``fft`` doesn't support keeping the ``pad_width`` attribute on the
    coordinates (at least for now).
    """
    da_padded = pad(sample_da_2d, pad_width, constant_values=0)
    da_fft = fft(da_padded)
    da_ifft = ifft(da_fft)
    da_unpadded = unpad(da_ifft, pad_width=pad_width)
    xrt.assert_allclose(sample_da_2d, da_unpadded)


@pytest.fixture
def sample_no_round_coords():
    """
    Define sample coordinates with no round floats
    """
    x = np.linspace(0, 10, 7)
    y = np.linspace(-4, 4, 13)
    coords = {
        "x": xr.DataArray(x, coords={"x": x}, dims=("x",)),
        "y": xr.DataArray(y, coords={"y": y}, dims=("y",)),
    }
    return coords


def test_pad_coordinates_no_round_coords(sample_no_round_coords):
    """
    Test _pad_coordinates on no round coordinates
    """
    padded_coords = _pad_coordinates(sample_no_round_coords, pad_width={"x": 3, "y": 4})
    assert padded_coords["x"].size == 13
    npt.assert_allclose(padded_coords["x"], np.linspace(-5, 15, 13))
    assert padded_coords["y"].size == 21
    npt.assert_allclose(padded_coords["y"], np.linspace(-4 - 8 / 3, 4 + 8 / 3, 21))


@pytest.fixture
def sample_da_2d_with_bad_coord():
    """
    Defines a 2D sample xarray.DataArray with a bad coordinate

    A bad coordinate is an additional coordinate that holds the same dimensions
    that will be used for padding.
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.arange(11 * 17, dtype=float).reshape(17, 11)
    height = z / 2  # bad coordinate
    dims = ("y", "x")
    coords = {"x": ("x", x), "y": ("y", y), "height": (dims, height)}
    return xr.DataArray(z, coords=coords, dims=dims)


@pytest.mark.parametrize(
    "pad_width",
    [{"x": 3}, {"y": 5}, {"x": 2, "y": 4}],
)
def test_pad_with_extra_1d_coordinate(pad_width, sample_da_2d_with_bad_coord):
    """
    Test pad on a grid that has an extra dimension
    """
    msg = (
        "Please, drop the following coordinates from the passed DataArray "
        + "before trying to pad it: 'height'."
    )
    with pytest.raises(ValueError, match=msg):
        pad(sample_da_2d_with_bad_coord, pad_width=pad_width)
