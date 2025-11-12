# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test ellipsoid classes.
"""

import re

import pytest

from harmonica import OblateEllipsoid, ProlateEllipsoid, TriaxialEllipsoid


class TestProlateEllipsoid:
    """Test the ProlateEllipsoid class."""

    @pytest.mark.parametrize(("a", "b"), [(35.0, 50.0), (50.0, 50.0)])
    def test_invalid_semiaxes(self, a, b):
        """Test error if not a > b."""
        msg = re.escape("Invalid ellipsoid axis lengths for prolate ellipsoid")
        with pytest.raises(ValueError, match=msg):
            ProlateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))

    def test_value_of_c(self):
        """Test if c is always equal to b."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
        assert ellipsoid.b == ellipsoid.c
        # Update the value of b and check again
        ellipsoid.b = 45.0
        assert ellipsoid.b == ellipsoid.c

    def test_roll_equal_to_zero(self):
        """Test if roll is always equal to zero."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
        assert ellipsoid.roll == 0.0


class TestOblateEllipsoid:
    """Test the OblateEllipsoid class."""

    @pytest.mark.parametrize(("a", "b"), [(50.0, 35.0), (50.0, 50.0)])
    def test_invalid_semiaxes(self, a, b):
        """Test error if not a < b."""
        msg = re.escape("Invalid ellipsoid axis lengths for oblate ellipsoid")
        with pytest.raises(ValueError, match=msg):
            OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))

    def test_value_of_c(self):
        """Test if c is always equal to b."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
        assert ellipsoid.b == ellipsoid.c
        # Update the value of b and check again
        ellipsoid.b = 45.0
        assert ellipsoid.b == ellipsoid.c

    def test_roll_equal_to_zero(self):
        """Test if roll is always equal to zero."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
        assert ellipsoid.roll == 0.0


class TestTriaxialEllipsoid:
    """Test the TriaxialEllipsoid class."""

    @pytest.mark.parametrize(
        ("a", "b", "c"), [(50.0, 35.0, 45.0), (50.0, 50.0, 50.0), (60.0, 50.0, 50.0)]
    )
    def test_invalid_semiaxes(self, a, b, c):
        """Test error if not a > b > c."""
        msg = re.escape("Invalid ellipsoid axis lengths for triaxial ellipsoid")
        with pytest.raises(ValueError, match=msg):
            TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, centre=(0, 0, 0))
