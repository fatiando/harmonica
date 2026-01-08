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
from unittest.mock import patch

import numpy as np
import pytest

from harmonica import Ellipsoid

try:
    import pyvista
except ImportError:
    pyvista = None


class TestEllipsoid:
    """Test the Ellipsoid class."""

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
            Ellipsoid(a, b, c)

    @pytest.mark.parametrize("semiaxis", ["a", "b", "c"])
    def test_non_positive_semiaxis_setter(self, semiaxis):
        """Test error after non-positive semiaxis when using the setter."""
        a, b, c = 50.0, 40.0, 35.0
        ellipsoid = Ellipsoid(a, b, c)
        msg = re.escape(f"Invalid value of '{semiaxis}' equal to '{-1.0}'")
        with pytest.raises(ValueError, match=msg):
            setattr(ellipsoid, semiaxis, -1.0)

    def test_semiaxes_setter(self):
        """Test setters for semiaxes."""
        a, b, c = 50.0, 40.0, 35.0
        ellipsoid = Ellipsoid(a, b, c)
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

    def test_rotation_matrix_of_sphere(self):
        """Make sure the rotation matrix of a sphere is always the identity."""
        a = 10.0
        sphere = Ellipsoid(a, a, a)
        np.testing.assert_array_equal(
            sphere.rotation_matrix, np.eye(3, dtype=np.float64)
        )


class TestPhysicalProperties:
    @pytest.fixture
    def semiaxes(self):
        """Ellipsoid semiaxes."""
        return 50.0, 35.0, 25.0

    def test_density(self, semiaxes):
        """
        Test assigning density to the ellipsoid.
        """
        a, b, c = semiaxes
        density = 3.0
        ellipsoid = Ellipsoid(a, b, c, density=density)
        assert ellipsoid.density == density
        # Check overwriting it
        new_density = -4.0
        ellipsoid.density = new_density
        assert ellipsoid.density == new_density
        # Check density as None
        ellipsoid = Ellipsoid(a, b, c, density=None)
        assert ellipsoid.density is None

    def test_invalid_density(self, semiaxes):
        """
        Test errors after invalid density.
        """
        a, b, c = semiaxes
        density = [1.0, 3.0]
        msg = re.escape("Invalid 'density' of type 'list'")
        with pytest.raises(TypeError, match=msg):
            Ellipsoid(a, b, c, density=density)

    @pytest.mark.parametrize("susceptibility", [0.1, "tensor", None])
    def test_susceptibility(self, semiaxes, susceptibility):
        """
        Test assigning susceptibility to the ellipsoid.
        """
        a, b, c = semiaxes
        if susceptibility == "tensor":
            susceptibility = np.random.default_rng(seed=42).uniform(size=(3, 3))
        ellipsoid = Ellipsoid(a, b, c, susceptibility=susceptibility)

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

    def test_invalid_susceptibility_type(self, semiaxes):
        """
        Test errors after invalid susceptibility type.
        """
        a, b, c = semiaxes
        susceptibility = [1.0, 3.0]
        msg = re.escape("Invalid 'susceptibility' of type 'list'")
        with pytest.raises(TypeError, match=msg):
            Ellipsoid(a, b, c, susceptibility=susceptibility)

    def test_invalid_susceptibility_shape(self, semiaxes):
        """
        Test errors after invalid susceptibility shape.
        """
        a, b, c = semiaxes
        susceptibility = np.array([[1, 2, 3], [4, 5, 6]])
        msg = re.escape("Invalid 'susceptibility' as an array with shape '(2, 3)'")
        with pytest.raises(ValueError, match=msg):
            Ellipsoid(a, b, c, susceptibility=susceptibility)

    @pytest.mark.parametrize("remanent_mag_type", ["array", "list", None])
    def test_remanent_mag(self, semiaxes, remanent_mag_type):
        """
        Test assigning susceptibility to the ellipsoid.
        """
        a, b, c = semiaxes
        if remanent_mag_type == "array":
            remanent_mag = np.array([1.0, 2.0, 3.0])
        elif remanent_mag_type == "list":
            remanent_mag = [1.0, 2.0, 3.0]
        else:
            remanent_mag = None

        ellipsoid = Ellipsoid(a, b, c, remanent_mag=remanent_mag)

        # Check if it was correctly assigned
        if remanent_mag is None:
            assert ellipsoid.remanent_mag is None
        else:
            np.testing.assert_almost_equal(ellipsoid.remanent_mag, remanent_mag)

        # Check overwriting it
        new_remanent_mag = np.array([-4.0, -5.0, 3.0])
        ellipsoid.remanent_mag = new_remanent_mag
        np.testing.assert_almost_equal(ellipsoid.remanent_mag, new_remanent_mag)

    def test_invalid_remanent_mag_type(self, semiaxes):
        """
        Test errors after invalid remanent_mag type.
        """

        class Dummy: ...

        a, b, c = semiaxes
        remanent_mag = Dummy()
        msg = re.escape("Invalid 'remanent_mag' of type 'Dummy'")
        with pytest.raises(TypeError, match=msg):
            Ellipsoid(a, b, c, remanent_mag=remanent_mag)

    def test_invalid_remanent_mag_shape(self, semiaxes):
        """
        Test errors after invalid remanent_mag shape.
        """
        a, b, c = semiaxes
        remanent_mag = np.array([1.0, 2.0])
        msg = re.escape("Invalid 'remanent_mag' with shape '(2,)'")
        with pytest.raises(ValueError, match=msg):
            Ellipsoid(a, b, c, remanent_mag=remanent_mag)


@pytest.mark.skipif(pyvista is None, reason="requires pyvista")
class TestToPyvista:
    """Test exporting ellipsoids to PyVista objects."""

    @pytest.fixture
    def ellipsoid(self):
        a, b, c = 3.0, 2.0, 1.0
        yaw, pitch, roll = 73.0, 14.0, -35.0
        center = (43.0, -72.0, 105)
        return Ellipsoid(a, b, c, yaw=yaw, pitch=pitch, roll=roll, center=center)

    @patch("harmonica._forward.ellipsoids.ellipsoids.pyvista", None)
    def test_pyvista_missing_error(self, ellipsoid):
        """
        Check if error is raised when pyvista is not installed.
        """
        with pytest.raises(ImportError):
            ellipsoid.to_pyvista()

    def test_pyvista_object(self, ellipsoid):
        """
        Check if method works as expected.
        """
        ellipsoid_pv = ellipsoid.to_pyvista()
        assert isinstance(ellipsoid_pv, pyvista.PolyData)
        # rtol needed since the parametric ellipsoid is not the exact surface.
        np.testing.assert_allclose(ellipsoid_pv.center, ellipsoid.center, rtol=1e-4)
        np.testing.assert_allclose(
            ellipsoid_pv.volume,
            4 / 3 * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c,
            rtol=1e-3,
        )


class TestString:
    """Test string representation of ellipsoids."""

    a, b, c = 3, 2, 1
    yaw, pitch, roll = 73, 14, -35
    center = (43.0, -72.0, 105)

    @pytest.fixture
    def ellipsoid(self):
        return Ellipsoid(
            self.a,
            self.b,
            self.c,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            center=self.center,
        )

    def test_triaxial(self, ellipsoid):
        expected = (
            "Ellipsoid:"
            "\n  • a:      3.0 m"
            "\n  • b:      2.0 m"
            "\n  • c:      1.0 m"
            "\n  • center: (43.0, -72.0, 105.0) m"
            "\n  • yaw:    73.0"
            "\n  • pitch:  14.0"
            "\n  • roll:   -35.0"
        )
        assert expected == str(ellipsoid)

    def test_density(self, ellipsoid):
        ellipsoid.density = -400
        (density_line,) = [
            line for line in str(ellipsoid).split("\n") if "density" in line
        ]
        expected_line = "  • density: -400.0 kg/m³"
        assert density_line == expected_line

    def test_susceptibility(self, ellipsoid):
        ellipsoid.susceptibility = 0.3
        (sus_line,) = [
            line for line in str(ellipsoid).splitlines() if "susceptibility" in line
        ]
        expected_line = "  • susceptibility: 0.3"
        assert sus_line == expected_line

    def test_remanent_mag(self, ellipsoid):
        ellipsoid.remanent_mag = (12, -43, 59)
        (rem_line,) = [
            line for line in str(ellipsoid).splitlines() if "remanent_mag" in line
        ]
        expected_line = "  • remanent_mag: (12.0, -43.0, 59.0) A/m"
        assert rem_line == expected_line

    def test_susceptibility_tensor(self, ellipsoid):
        sus = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        ellipsoid.susceptibility = sus
        # Grab the susceptibility tensor lines
        matrix_lines, record = [], False
        for line in str(ellipsoid).splitlines():
            if "susceptibility" in line:
                record = True
                continue
            if record:
                matrix_lines.append(line)
            if len(matrix_lines) == 3:
                break

        expected = [8 * " " + line for line in str(sus).splitlines()]
        assert matrix_lines == expected


class TestRepr:
    """Test the ``Ellipsoid.__repr__`` method."""

    a, b, c = 3, 2, 1
    yaw, pitch, roll = 73, 14, -35
    center = (43.0, -72.0, 105)

    @pytest.fixture
    def ellipsoid(self):
        return Ellipsoid(
            self.a,
            self.b,
            self.c,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            center=self.center,
        )

    def test_triaxial(self, ellipsoid):
        expected = (
            "harmonica.Ellipsoid("
            "a=3.0, b=2.0, c=1.0, center=(43.0, -72.0, 105.0), "
            "yaw=73.0, pitch=14.0, roll=-35.0"
            ")"
        )
        assert expected == repr(ellipsoid)

    def test_density(self, ellipsoid):
        ellipsoid.density = -400
        expected = (
            "harmonica.Ellipsoid("
            "a=3.0, b=2.0, c=1.0, center=(43.0, -72.0, 105.0), "
            "yaw=73.0, pitch=14.0, roll=-35.0, density=-400.0"
            ")"
        )
        assert expected == repr(ellipsoid)

    def test_susceptibility(self, ellipsoid):
        ellipsoid.susceptibility = 0.3
        expected = (
            "harmonica.Ellipsoid("
            "a=3.0, b=2.0, c=1.0, center=(43.0, -72.0, 105.0), "
            "yaw=73.0, pitch=14.0, roll=-35.0, susceptibility=0.3"
            ")"
        )
        assert expected == repr(ellipsoid)

    def test_remanent_mag(self, ellipsoid):
        ellipsoid.remanent_mag = (12, -43, 59)
        expected = (
            "harmonica.Ellipsoid("
            "a=3.0, b=2.0, c=1.0, center=(43.0, -72.0, 105.0), "
            "yaw=73.0, pitch=14.0, roll=-35.0, remanent_mag=[ 12 -43  59]"
            ")"
        )
        assert expected == repr(ellipsoid)

    def test_susceptibility_tensor(self, ellipsoid):
        sus = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        ellipsoid.susceptibility = sus
        expected = (
            "harmonica.Ellipsoid("
            "a=3.0, b=2.0, c=1.0, center=(43.0, -72.0, 105.0), "
            "yaw=73.0, pitch=14.0, roll=-35.0, "
            "susceptibility=[[1 0 0] [0 2 0] [0 0 3]]"
            ")"
        )
        assert expected == repr(ellipsoid)
