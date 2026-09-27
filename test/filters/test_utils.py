# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test utility functions for the filters submodule.
"""

import re

import bordado as bd
import numpy as np
import pytest
import xarray as xr

from harmonica.filters._utils import get_spacing


class TestGetSpacing:
    """Test the ``get_spacing`` private function."""

    def test_get_spacing(self):
        spacing = 2.3
        x = bd.line_coordinates(-2.0, 8.0, spacing=spacing, adjust="region")
        coordinate = xr.DataArray(x)
        np.testing.assert_allclose(spacing, get_spacing(coordinate))

    def test_not_evenly_spaced(self):
        coordinate = xr.DataArray([1.0, 2.0, 4.0, 5.0])
        msg = re.escape(
            f"Invalid '{coordinate.name}' coordinates: they must be evenly spaced."
        )
        with pytest.raises(ValueError, match=msg):
            get_spacing(coordinate)

    def test_not_ordered(self):
        spacing = 2.3
        x = bd.line_coordinates(-2.0, 8.0, spacing=spacing, adjust="region")[::-1]
        coordinate = xr.DataArray(x)
        msg = re.escape(
            f"Invalid coordinate '{coordinate.name}': it must be increasingly ordered."
        )
        with pytest.raises(ValueError, match=msg):
            get_spacing(coordinate)
