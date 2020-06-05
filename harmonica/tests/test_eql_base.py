"""
Test the EQLBase class
"""
import warnings
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd
import verde.base as vdb

from ..equivalent_layer.base import BaseEQL, _pop_extra_coords


def test_pop_extra_coords():
    """
    Test _pop_extra_coords private function
    """
    # Check if extra_coords is removed from kwargs
    kwargs = {"bla": 1, "blabla": 2, "extra_coords": 1400.0}
    with warnings.catch_warnings(record=True) as warn:
        _pop_extra_coords(kwargs)
        assert len(warn) == 1
        assert issubclass(warn[0].category, UserWarning)
    assert "extra_coords" not in kwargs

    # Check if kwargs is not touched if no extra_coords are present
    kwargs = {"bla": 1, "blabla": 2}
    _pop_extra_coords(kwargs)
    assert kwargs == {"bla": 1, "blabla": 2}
