# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test FFT functions.
"""

import re

import bordado as bd
import numpy as np
import pytest
import xarray as xr
import xrft

from harmonica.filters import fft, ifft
from harmonica.filters._fft import (
    _fftfreq,
    _get_dimensional_coordinate,
    _get_spacing,
    _ifftfreq,
)


@pytest.fixture
def synthetic_grid():
    """
    Synthetic 2D grid in space domain.
    """
    # Create easting and northing coordinates with a static shift
    easting = np.linspace(-2, 8, 21)
    northing = np.linspace(4, 15, 41)
    xx, yy = np.meshgrid(easting, northing)
    z = np.cos(xx) * np.sin(yy)
    dims = ("northing", "easting")
    coords = {"easting": easting, "northing": northing}
    return xr.DataArray(z, coords=coords, dims=dims)


def rename_coordinate(da: xr.DataArray, renaming_dict: dict[str, str]) -> xr.DataArray:
    """
    Rename coordinate of xarray.DataArray without modifying the dimension.
    """
    for old_coord, new_coord in renaming_dict.items():
        (dim,) = getattr(da, old_coord).dims
        da = da.assign_coords({new_coord: (dim, getattr(da, old_coord).values)})
        da = da.drop_vars(old_coord)
    return da


def test_round_loop(synthetic_grid):
    """
    Test if applying FFT and iFFT results in the original array.

    Test the normal FFT and iFFT operations considering true phase and true amplitude.
    It also checks that the static shift in the original coordiantes is restored.
    """
    recovered = ifft(fft(synthetic_grid))
    xr.testing.assert_allclose(synthetic_grid, recovered)


class TestPrefix:
    """
    Test if prefix of frequency coordinates is correctly set.
    """

    def test_fft_prefix(self, synthetic_grid):
        """
        Test custom prefix to fft.
        """
        prefix = "new_prefix_"
        fft_grid = fft(synthetic_grid, prefix=prefix)
        assert fft_grid.dims == (f"{prefix}northing", f"{prefix}easting")
        assert f"{prefix}easting" in fft_grid.coords
        assert f"{prefix}northing" in fft_grid.coords

    def test_ifft_prefix(self, synthetic_grid):
        """
        Test custom prefix to ifft.
        """
        prefix = "new_prefix_"
        fft_grid = fft(synthetic_grid, prefix=prefix)
        ifft_grid = ifft(fft_grid, prefix=prefix)
        assert ifft_grid.dims == ("northing", "easting")
        assert "easting" in ifft_grid.coords
        assert "northing" in ifft_grid.coords


class TestErrors:
    """
    Test sanity errors raised by ``fft`` and ``ifft``.
    """

    def test_ifft_prefix_error_dimension(self, synthetic_grid):
        """
        Test error raised after prefix in dimension not found.
        """
        prefix = "new_prefix_"
        fft_grid = fft(synthetic_grid, prefix=prefix)
        msg = re.escape("Invalid frequency dimension")
        with pytest.raises(ValueError, match=msg):
            ifft(fft_grid)

    @pytest.mark.parametrize("coordinate", ["easting", "northing"])
    def test_ifft_prefix_error_coordiante(self, synthetic_grid, coordinate):
        """
        Test error raised after prefix in frequency coordinates not found.
        """
        fft_grid = fft(synthetic_grid)
        fft_grid = rename_coordinate(
            fft_grid, {f"freq_{coordinate}": f"blah_{coordinate}"}
        )
        msg = re.escape(f"Invalid dimensional coordinate 'blah_{coordinate}'")
        with pytest.raises(ValueError, match=msg):
            ifft(fft_grid)

    @pytest.mark.parametrize("fft_func", [fft, ifft])
    def test_not_xarray(self, fft_func):
        """Test error if passed grid is not a ``xarray.DataArray``."""
        grid = np.arange(25).reshape(5, 5)
        msg = re.escape(
            f"Invalid 'grid' of type '{type(grid).__name__}'. "
            "It must be an xarray.DataArray."
        )
        with pytest.raises(TypeError, match=msg):
            fft_func(grid)

    @pytest.mark.parametrize("fft_func", [fft, ifft])
    def test_not_2d_grid(self, fft_func):
        """Test error if passed grid is not 2D."""
        x = np.arange(10)
        z = np.random.default_rng(seed=42).uniform(size=x.size)
        da = xr.DataArray(z, coords={"x": x})
        msg = re.escape("Invalid grid array with '1' dimension. It must be a 2D array.")
        with pytest.raises(ValueError, match=msg):
            fft_func(da)


class TestAgainstXRFT:
    """
    Compare our FFT results against the ones obtained with ``xrft``.

    .. note::

        We should replace these tests with proper tests against analytical solutions to
        stop depending in ``xrft`` also to run tests.
    """

    def test_fft(self, synthetic_grid):
        """
        Test FFT results.
        """
        fft_hm = fft(synthetic_grid)
        fft_xrft = xrft.fft(synthetic_grid)
        xr.testing.assert_allclose(fft_hm, fft_xrft)

    @pytest.mark.filterwarnings("ignore:Default ifft's behaviour")
    def test_ifft(self, synthetic_grid):
        """
        Test FFT results.
        """
        recovered_hm = ifft(fft(synthetic_grid))
        recovered_xrft = xrft.ifft(xrft.fft(synthetic_grid))
        xr.testing.assert_allclose(recovered_hm, recovered_xrft)


class TestGetSpacing:
    """Test the ``_get_spacing`` private function."""

    def test_get_spacing(self):
        spacing = 2.3
        x = bd.line_coordinates(-2.0, 8.0, spacing=spacing, adjust="region")
        coordinate = xr.DataArray(x)
        np.testing.assert_allclose(spacing, _get_spacing(coordinate))

    def test_not_evenly_spaced(self):
        coordinate = xr.DataArray([1.0, 2.0, 4.0, 5.0])
        msg = re.escape(
            f"Invalid '{coordinate.name}' coordinates: they must be evenly spaced."
        )
        with pytest.raises(ValueError, match=msg):
            _get_spacing(coordinate)

    def test_not_ordered(self):
        spacing = 2.3
        x = bd.line_coordinates(-2.0, 8.0, spacing=spacing, adjust="region")[::-1]
        coordinate = xr.DataArray(x)
        msg = re.escape(
            f"Invalid coordinate '{coordinate.name}': it must be increasingly ordered."
        )
        with pytest.raises(ValueError, match=msg):
            _get_spacing(coordinate)


class TestDimensionalCoordinate:
    """Test the ``_get_dimensional_coordinate`` private function."""

    @pytest.mark.parametrize("dim", ["easting", "northing"])
    def test_get_dimensional_coordinate(self, synthetic_grid, dim):
        """Test getting the dimensional coordinate assigned with a particular dim."""
        # Rename the coordiante to make the test less trivial
        new_coord_name = "blah"
        grid = rename_coordinate(synthetic_grid, {dim: new_coord_name})
        assert _get_dimensional_coordinate(grid, dim) == new_coord_name

    def test_no_dimensional_coordinate(self, synthetic_grid):
        """Test error if no dimensional coordinate is found."""
        dim = "blah"
        msg = re.escape(f"Couldn't find dimensional coordinate for dimension '{dim}'.")
        with pytest.raises(ValueError, match=msg):
            _get_dimensional_coordinate(synthetic_grid, dim)

    @pytest.mark.parametrize("dim", ["easting", "northing"])
    def test_multiple_dimensional_coordinates(self, synthetic_grid, dim):
        """Test error multiple dimensional coordinates are found."""
        bad_coord = "bad-coord"
        synthetic_grid = synthetic_grid.assign_coords(
            {bad_coord: (dim, getattr(synthetic_grid, dim).values)}
        )
        bad_coords = f"{dim}, {bad_coord}"
        msg = re.escape(
            f"Multiple dimensional coordinates ({bad_coords}) found "
            f"for the '{dim}' dimension. "
            "Leave only one dimensional coordinate per dimension."
        )
        with pytest.raises(ValueError, match=msg):
            _get_dimensional_coordinate(synthetic_grid, dim)


class TestFFTFreq:
    """Test the ``_fftfreq`` and ``_ifftfreq`` private functions."""

    default_spacing = 0.5

    @pytest.mark.parametrize("spacing", [None, default_spacing])
    def test_fftfreq(self, spacing):
        coord = xr.DataArray(
            bd.line_coordinates(
                -4.0, 11.0, spacing=self.default_spacing, adjust="region"
            )
        )
        freq = _fftfreq(coord, spacing=spacing)
        # Check if frequencies are evenly spaced
        freq_spacing = freq[1] - freq[0]
        np.testing.assert_allclose(freq_spacing, freq[1:] - freq[:-1])
        # Check if frequencies are sorted
        assert np.all(freq[1:] > freq[:-1])

    @pytest.mark.parametrize("spacing", [None, default_spacing])
    def test_ifftfreq(self, spacing):
        freq = xr.DataArray(
            bd.line_coordinates(
                -4.0, 11.0, spacing=self.default_spacing, adjust="region"
            )
        )
        coord = _ifftfreq(freq, spacing=spacing)
        # Check if frequencies are evenly spaced
        coord_spacing = coord[1] - coord[0]
        np.testing.assert_allclose(coord_spacing, coord[1:] - coord[:-1])
        # Check if frequencies are sorted
        assert np.all(coord[1:] > coord[:-1])

    def test_roundtrip(self):
        # Define spatial coordinates
        coord = xr.DataArray(
            bd.line_coordinates(
                22.0, 32.0, spacing=self.default_spacing, adjust="region"
            )
        )
        # Define frequencies and add shift
        freq = xr.DataArray(_fftfreq(coord))
        freq.attrs.update({"shift": coord.values.min()})
        # Check that the recovered spatial coordinates are close to the coords
        recovered = _ifftfreq(freq)
        np.testing.assert_allclose(recovered, coord)

    def test_ifftfreq_no_shift(self):
        """
        Test ``_ifftfreq`` when the frequency coordinates have no **shift** attr.
        """
        # Define spatial coordinates
        coord = xr.DataArray(
            bd.line_coordinates(
                22.0, 32.0, spacing=self.default_spacing, adjust="region"
            )
        )
        # Define frequencies without shift
        freq = xr.DataArray(_fftfreq(coord))
        recovered = _ifftfreq(freq)
        # Check that the recovered doesn't match the original coord
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(recovered, coord)
        # Check that the recovered are centered around zero
        assert recovered[0] == -recovered[-1]

    @pytest.mark.parametrize("func", [_fftfreq, _ifftfreq])
    def test_invalid_coordiante(self, func):
        """Test error if coordinate is not 1D."""
        coord = xr.DataArray(np.arange(25).reshape(5, 5))
        with pytest.raises(ValueError, match="It must be 1D"):
            func(coord)
