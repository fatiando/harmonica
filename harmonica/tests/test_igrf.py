# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the IGRF results against the ones calculated by the BGS.
"""
import os

import pytest
import xarray as xr

from .._spherical_harmonics.igrf import load_igrf, fetch_igrf14,
interpolate_coefficients, IGRF14


MODULE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MODULE_DIR, "data")


def test_fetch_igrf14():
    "Check that the coefficient file can be fetched from Zenodo"
    fname = fetch_igrf14()
    assert os.path.exists(fname)


def test_load_igrf_shapes():
    "Check if things read have the right shapes and sizes"
    g, h, g_sv, h_sv, years = load_igrf(fetch_igrf14())
    assert years.size == 26
    assert np.allclose(years, np.arange(1900, 2026, 5))
    assert g.shape == (13, 26)
    assert h.shape == (13, 26)
    assert g_sv.size == 13
    assert h_sv.size == 13


def test_load_igrf_file_not_found():
    "Check if it fails when given a bad file name"
    with pytest.raises(IOError):
        load_igrf("bla.slkdjsldjh")


def test_igrf14_bgs():
    "Check against files calculated by the BGS service"
    bgs = xr.load_dataset()
