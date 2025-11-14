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

import numpy as np
import pytest

from harmonica import OblateEllipsoid, ProlateEllipsoid, TriaxialEllipsoid


class TestProlateEllipsoid:
    """Test the ProlateEllipsoid class."""

    @pytest.mark.parametrize(("a", "b"), [(35.0, 50.0), (50.0, 50.0)])
    def test_invalid_semiaxes(self, a, b):
        """Test error if not a > b."""
        msg = re.escape("Invalid ellipsoid axis lengths for prolate ellipsoid")
        with pytest.raises(ValueError, match=msg):
            ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))

    @pytest.mark.parametrize("semiaxis", ["a", "b"])
    def test_non_positive_semiaxis(self, semiaxis):
        """Test error after non-positive semiaxis."""
        match semiaxis:
            case "a":
                a, b = -1.0, 40.0
            case "b":
                a, b = 50.0, -1.0
            case _:
                raise ValueError()
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))

    @pytest.mark.parametrize("semiaxis", ["a", "b"])
    def test_non_positive_semiaxis_setter(self, semiaxis):
        """Test error after non-positive semiaxis when using the setter."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            setattr(ellipsoid, semiaxis, -1.0)

    def test_semiaxes_setter(self):
        """Test setters for semiaxes."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        # Test setter of a
        new_a = a + 1
        ellipsoid.a = new_a
        assert ellipsoid.a == new_a
        # Test setter of b
        new_b = b + 1
        ellipsoid.b = new_b
        assert ellipsoid.b == new_b

    def test_invalid_semiaxes_setter(self):
        """Test error if not a > b when using the setter."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        msg = re.escape("Invalid ellipsoid axis lengths for prolate ellipsoid")
        with pytest.raises(ValueError, match=msg):
            ellipsoid.a = 20.0
        with pytest.raises(ValueError, match=msg):
            ellipsoid.b = 70.0

    def test_value_of_c(self):
        """Test if c is always equal to b."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        assert ellipsoid.b == ellipsoid.c
        # Update the value of b and check again
        ellipsoid.b = 45.0
        assert ellipsoid.b == ellipsoid.c

    def test_roll_equal_to_zero(self):
        """Test if roll is always equal to zero."""
        a, b = 50.0, 35.0
        ellipsoid = ProlateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        assert ellipsoid.roll == 0.0


class TestOblateEllipsoid:
    """Test the OblateEllipsoid class."""

    @pytest.mark.parametrize(("a", "b"), [(50.0, 35.0), (50.0, 50.0)])
    def test_invalid_semiaxes(self, a, b):
        """Test error if not a < b."""
        msg = re.escape("Invalid ellipsoid axis lengths for oblate ellipsoid")
        with pytest.raises(ValueError, match=msg):
            OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))

    @pytest.mark.parametrize("semiaxis", ["a", "b"])
    def test_non_positive_semiaxis(self, semiaxis):
        """Test error after non-positive semiaxis."""
        match semiaxis:
            case "a":
                a, b = -1.0, 40.0
            case "b":
                a, b = 50.0, -1.0
            case _:
                raise ValueError()
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))

    @pytest.mark.parametrize("semiaxis", ["a", "b"])
    def test_non_positive_semiaxis_setter(self, semiaxis):
        """Test error after non-positive semiaxis when using the setter."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            setattr(ellipsoid, semiaxis, -1.0)

    def test_semiaxes_setter(self):
        """Test setters for semiaxes."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        # Test setter of a
        new_a = a + 1
        ellipsoid.a = new_a
        assert ellipsoid.a == new_a
        # Test setter of b
        new_b = b + 1
        ellipsoid.b = new_b
        assert ellipsoid.b == new_b

    def test_invalid_semiaxes_setter(self):
        """Test error if not a < b when using the setter."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        msg = re.escape("Invalid ellipsoid axis lengths for oblate ellipsoid")
        with pytest.raises(ValueError, match=msg):
            ellipsoid.a = 70.0
        with pytest.raises(ValueError, match=msg):
            ellipsoid.b = 20.0

    def test_value_of_c(self):
        """Test if c is always equal to b."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
        assert ellipsoid.b == ellipsoid.c
        # Update the value of b and check again
        ellipsoid.b = 45.0
        assert ellipsoid.b == ellipsoid.c

    def test_roll_equal_to_zero(self):
        """Test if roll is always equal to zero."""
        a, b = 35.0, 50.0
        ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, center=(0, 0, 0))
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
            TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0))

    @pytest.mark.parametrize("semiaxis", ["a", "b", "c"])
    def test_non_positive_semiaxis(self, semiaxis):
        """Test error after non-positive semiaxis."""
        match semiaxis:
            case "a":
                a, b, c = -1.0, 40.0, 35.0
            case "b":
                a, b, c = 50.0, -1.0, 35.0
            case "c":
                a, b, c = 50.0, 40.0, -1.0
            case _:
                raise ValueError()
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0))

    @pytest.mark.parametrize("semiaxis", ["a", "b", "c"])
    def test_non_positive_semiaxis_setter(self, semiaxis):
        """Test error after non-positive semiaxis when using the setter."""
        a, b, c = 50.0, 40.0, 35.0
        ellipsoid = TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0))
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            setattr(ellipsoid, semiaxis, -1.0)

    def test_semiaxes_setter(self):
        """Test setters for semiaxes."""
        a, b, c = 50.0, 40.0, 35.0
        ellipsoid = TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0))
        # Test setter of a
        new_a = a + 1
        ellipsoid.a = new_a
        assert ellipsoid.a == new_a
        # Test setter of b
        new_b = b + 1
        ellipsoid.b = new_b
        assert ellipsoid.b == new_b
        # Test setter of c
        new_c = c + 1
        ellipsoid.c = new_c
        assert ellipsoid.c == new_c

    def test_invalid_semiaxes_setter(self):
        """Test error if not a > b > c when using the setter."""
        a, b, c = 50.0, 40.0, 30.0
        ellipsoid = TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0))
        msg = re.escape("Invalid ellipsoid axis lengths for triaxial ellipsoid")
        with pytest.raises(ValueError, match=msg):
            ellipsoid.a = 30.0
        with pytest.raises(ValueError, match=msg):
            ellipsoid.b = 20.0
        with pytest.raises(ValueError, match=msg):
            ellipsoid.c = 70.0


@pytest.mark.parametrize(
    "ellipsoid_class", [OblateEllipsoid, ProlateEllipsoid, TriaxialEllipsoid]
)
class TestPhysicalProperties:
    @pytest.fixture
    def ellipsoid_args(self, ellipsoid_class):
        if ellipsoid_class is OblateEllipsoid:
            args = {
                "a": 20.0,
                "b": 50.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "center": (0, 0, 0),
            }
        elif ellipsoid_class is ProlateEllipsoid:
            args = {
                "a": 50.0,
                "b": 20.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "center": (0, 0, 0),
            }
        elif ellipsoid_class is TriaxialEllipsoid:
            args = {
                "a": 50.0,
                "b": 20.0,
                "c": 10.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "center": (0, 0, 0),
            }
        else:
            raise TypeError()
        return args

    def test_density(self, ellipsoid_class, ellipsoid_args):
        """
        Test assigning density to the ellipsoid.
        """
        density = 3.0
        ellipsoid = ellipsoid_class(**ellipsoid_args, density=density)
        assert ellipsoid.density == density
        # Check overwriting it
        new_density = -4.0
        ellipsoid.density = new_density
        assert ellipsoid.density == new_density
        # Check density as None
        ellipsoid = ellipsoid_class(**ellipsoid_args, density=None)
        assert ellipsoid.density is None

    def test_invalid_density(self, ellipsoid_class, ellipsoid_args):
        """
        Test errors after invalid density.
        """
        density = [1.0, 3.0]
        msg = re.escape("Invalid 'density' of type 'list'")
        with pytest.raises(TypeError, match=msg):
            ellipsoid_class(**ellipsoid_args, density=density)

    @pytest.mark.parametrize("susceptibility", [0.1, "tensor", None])
    def test_susceptibility(self, ellipsoid_class, ellipsoid_args, susceptibility):
        """
        Test assigning susceptibility to the ellipsoid.
        """
        if susceptibility == "tensor":
            susceptibility = np.random.default_rng(seed=42).uniform(size=(3, 3))
        ellipsoid = ellipsoid_class(**ellipsoid_args, susceptibility=susceptibility)

        # Check if it was correctly assigned
        if susceptibility is None:
            assert ellipsoid.susceptibility is None
        elif isinstance(susceptibility, float):
            assert ellipsoid.susceptibility == susceptibility
        else:
            np.testing.assert_almost_equal(ellipsoid.susceptibility, susceptibility)

        # Check overwriting it
        new_sus = -4.0
        ellipsoid.susceptibility = new_sus
        assert ellipsoid.susceptibility == new_sus

    def test_invalid_susceptibility_type(self, ellipsoid_class, ellipsoid_args):
        """
        Test errors after invalid susceptibility type.
        """
        susceptibility = [1.0, 3.0]
        msg = re.escape("Invalid 'susceptibility' of type 'list'")
        with pytest.raises(TypeError, match=msg):
            ellipsoid_class(**ellipsoid_args, susceptibility=susceptibility)

    def test_invalid_susceptibility_shape(self, ellipsoid_class, ellipsoid_args):
        """
        Test errors after invalid susceptibility shape.
        """
        susceptibility = np.array([[1, 2, 3], [4, 5, 6]])
        msg = re.escape("Invalid 'susceptibility' as an array with shape '(2, 3)'")
        with pytest.raises(ValueError, match=msg):
            ellipsoid_class(**ellipsoid_args, susceptibility=susceptibility)

    @pytest.mark.parametrize("remanent_mag_type", ["array", "list", None])
    def test_remanent_mag(self, ellipsoid_class, ellipsoid_args, remanent_mag_type):
        """
        Test assigning susceptibility to the ellipsoid.
        """
        if remanent_mag_type == "array":
            remanent_mag = np.array([1.0, 2.0, 3.0])
        elif remanent_mag_type == "list":
            remanent_mag = [1.0, 2.0, 3.0]
        else:
            remanent_mag = None

        ellipsoid = ellipsoid_class(**ellipsoid_args, remanent_mag=remanent_mag)

        # Check if it was correctly assigned
        if remanent_mag is None:
            assert ellipsoid.remanent_mag is None
        else:
            np.testing.assert_almost_equal(ellipsoid.remanent_mag, remanent_mag)

        # Check overwriting it
        new_remanent_mag = np.array([-4.0, -5.0, 3.0])
        ellipsoid.remanent_mag = new_remanent_mag
        np.testing.assert_almost_equal(ellipsoid.remanent_mag, new_remanent_mag)

    def test_invalid_remanent_mag_type(self, ellipsoid_class, ellipsoid_args):
        """
        Test errors after invalid remanent_mag type.
        """

        class Dummy: ...

        remanent_mag = Dummy()
        msg = re.escape("Invalid 'remanent_mag' of type 'Dummy'")
        with pytest.raises(TypeError, match=msg):
            ellipsoid_class(**ellipsoid_args, remanent_mag=remanent_mag)

    def test_invalid_remanent_mag_shape(self, ellipsoid_class, ellipsoid_args):
        """
        Test errors after invalid remanent_mag shape.
        """
        remanent_mag = np.array([1.0, 2.0])
        msg = re.escape("Invalid 'remanent_mag' with shape '(2,)'")
        with pytest.raises(ValueError, match=msg):
            ellipsoid_class(**ellipsoid_args, remanent_mag=remanent_mag)
