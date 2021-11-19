# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test equivalent sources utility functions
"""
import warnings

import numpy as np
import pytest
from verde import scatter_points

from ..equivalent_sources.utils import cast_fit_input, pop_extra_coords


def test_pop_extra_coords():
    """
    Test pop_extra_coords private function
    """
    # Check if extra_coords is removed from kwargs
    kwargs = {"bla": 1, "blabla": 2, "extra_coords": 1400.0}
    with warnings.catch_warnings(record=True) as warn:
        pop_extra_coords(kwargs)
        assert len(warn) == 1
        assert issubclass(warn[0].category, UserWarning)
    assert "extra_coords" not in kwargs

    # Check if kwargs is not touched if no extra_coords are present
    kwargs = {"bla": 1, "blabla": 2}
    pop_extra_coords(kwargs)
    assert kwargs == {"bla": 1, "blabla": 2}


@pytest.mark.parametrize("weights_none", (False, True))
@pytest.mark.parametrize("dtype", ("float64", "float32", "int32", "int64"))
def test_cast_fit_input(weights_none, dtype):
    """
    Test cast_fit_input function
    """
    region = (-7e3, 4e3, 10e3, 25e3)
    coordinates = scatter_points(region=region, size=100, random_state=42)
    data = np.arange(coordinates[0].size, dtype="float64")
    if weights_none:
        weights = None
    else:
        weights = np.ones_like(data)
    coordinates, data, weights = cast_fit_input(coordinates, data, weights, dtype)
    # Check dtype of the outputs
    for coord in coordinates:
        assert coord.dtype == np.dtype(dtype)
    assert data.dtype == np.dtype(dtype)
    if weights_none:
        assert weights is None
    else:
        assert weights.dtype == np.dtype(dtype)
