# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import pytest
import verde as vd

from harmonica import point_gravity

from .._forward.create_ellipsoid import (
    OblateEllipsoid,
    ProlateEllipsoid,
    TriaxialEllipsoid,
)
from .._forward.ellipsoid_gravity import (
    ellipsoid_gravity,
)


def test_degenerate_ellipsoid_cases():
    """
    Test cases where the ellipsoid axes lengths are close to the boundary of
    accepted values.

    """
    # ellipsoids take (a, b, #c, yaw, pitch, #roll, centre)
    tri = TriaxialEllipsoid(5, 4.99999999, 4.99999998, 0, 0, 0, (0, 0, 0))
    pro = ProlateEllipsoid(5, 4.99999999, 0, 0, (0, 0, 0))
    obl = OblateEllipsoid(4.99999999, 5, 0, 0, (0, 0, 0))
    density = 2000
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    _, _, gu1 = ellipsoid_gravity(coordinates, tri, density, field="g")
    _, _, gu2 = ellipsoid_gravity(coordinates, pro, density, field="g")
    _, _, gu3 = ellipsoid_gravity(coordinates, obl, density, field="g")


def test_opposite_planes():
    """

    Test two surfaces produce the same anomaly but 'flipped' when including a
    rotation in the ellipsoid.

    """
    a, b, c = (4, 3, 2)  # triaxial ellipsoid
    yaw = 90
    pitch = 0
    roll = 0
    triaxial_example = TriaxialEllipsoid(a, b, c, yaw, pitch, roll, (0, 0, 0))
    density = 2000

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates1 = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )
    coordinates2 = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=-5
    )

    _, _, gu1 = ellipsoid_gravity(coordinates1, triaxial_example, density, field="g")
    _, _, gu2 = ellipsoid_gravity(coordinates2, triaxial_example, density, field="g")
    np.testing.assert_allclose(gu1, -np.flip(gu2))


def test_int_ext_boundary():
    """

    Check that the boundary of the internal and eternal components of the
    calculation align.

    """

    # compare a set value apart
    a, b, c = (5, 4, 3)
    ellipsoid = TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0, centre=(0, 0, 0))

    e = np.array([[4.9999999, 5.00000001]])
    n = np.array([[0.0, 0.0]])
    u = np.array([[0.0, 0.0]])
    coordinates = (e, n, u)

    ge, gn, gu = ellipsoid_gravity(coordinates, ellipsoid, 2000, field="g")

    np.testing.assert_allclose(ge[0, 0], ge[0, 1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gn[0, 0], gn[0, 1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gu[0, 0], gu[0, 1], rtol=1e-5, atol=1e-5)


class TestSymmetry:
    """
    Test symmetry in gravity fields.
    """

    def build_ellipsoid(self, ellipsoid_type):
        """
        Build sample ellipsoid.

        Use this function only to build a particular ellipsoid. Use the ``ellipsoid``
        fixture instead.
        """
        centre = (0, 0, 0)
        if ellipsoid_type == "triaxial":
            a, b, c = 3.2, 2.1, 1.3
            ellipsoid = TriaxialEllipsoid(
                a=a, b=b, c=c, yaw=0, pitch=0, roll=0, centre=centre
            )
        elif ellipsoid_type == "prolate":
            a, b = 3.2, 2.1
            ellipsoid = ProlateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
        elif ellipsoid_type == "oblate":
            a, b = 2.2, 3.1
            ellipsoid = OblateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
        else:
            msg = f"Invalid ellipsoid type: {ellipsoid_type}"
            raise ValueError(msg)
        return ellipsoid

    @pytest.fixture(params=["triaxial", "prolate", "oblate"])
    def ellipsoid(self, request):
        """
        Sample ellipsoid.
        """
        ellipsoid_type = request.param
        return self.build_ellipsoid(ellipsoid_type)

    def test_vertical_symmetry_on_surface(self, ellipsoid):
        """
        Test symmetry of gz across a vertical axis that passes through the center of the
        ellipsoid.
        """
        points = [(0, 0, ellipsoid.c), (0, 0, -ellipsoid.c)]
        density = 200
        gu_up, gu_down = tuple(
            ellipsoid_gravity(p, ellipsoid, density, field="u") for p in points
        )
        np.testing.assert_allclose(gu_up, -gu_down)

    @pytest.mark.parametrize("ellipsoid_type", ["oblate", "prolate"])
    @pytest.mark.parametrize("points", ["internal", "surface", "external"])
    def test_symmetry_on_circle(self, points, ellipsoid_type):
        """
        Test symmetry of |g| on circle around center of a prolate and oblate ellipsoid.

        Define a circle in the northing-upward plane, compute |g| on points along that
        circle. All values of |g| should be equal.
        """
        ellipsoid = self.build_ellipsoid(ellipsoid_type)

        # Build coordinates along circle centered in the center of the ellipsoid
        radius = ellipsoid.b
        if points == "internal":
            radius *= 0.5
        elif points == "external":
            radius *= 2
        theta = np.linspace(0, 2 * np.pi, 61)

        northing = radius * np.cos(theta)
        upward = radius * np.sin(theta)
        easting = np.zeros_like(northing)
        coordinates = (easting, northing, upward)

        # Compute gravity acceleration along the circle
        density = 200
        ge, gn, gu = ellipsoid_gravity(coordinates, ellipsoid, density, field="g")
        g = np.sqrt(ge**2 + gn**2 + gu**2)

        # Check that |g| is constant in the circle
        np.testing.assert_allclose(g[0], g)


class TestEllipsoidVsPointSource:
    """
    Test if gravity field of ellipsoids approximates the one of a point source.

    In the infinity limit, the gravity field of the ellipsoid should be the same as the
    one for a point source.
    """

    @pytest.fixture(params=["triaxial", "prolate", "oblate"])
    def ellipsoid(self, request):
        """
        Sample ellipsoid.
        """
        ellipsoid_type = request.param

        centre = (0, 0, 0)
        if ellipsoid_type == "triaxial":
            a, b, c = 3.2, 2.1, 1.3
            ellipsoid = TriaxialEllipsoid(
                a=a, b=b, c=c, yaw=0, pitch=0, roll=0, centre=centre
            )
        elif ellipsoid_type == "prolate":
            a, b = 3.2, 2.1
            ellipsoid = ProlateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
        elif ellipsoid_type == "oblate":
            a, b = 2.2, 3.1
            ellipsoid = OblateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
        else:
            msg = f"Invalid ellipsoid type: {ellipsoid_type}"
            raise ValueError(msg)
        return ellipsoid

    def test_approximation(self, ellipsoid):
        """
        Compare gravity field of ellipsoid with point source at large distance.

        The two fields should be close enough at a sufficient large distance.
        """
        phi, theta = 48.9, 12.3
        radius = max((ellipsoid.a, ellipsoid.b, ellipsoid.c)) * 1e3  # large radius
        coordinates = (
            radius * np.cos(phi) * np.cos(theta),
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(theta),
        )
        density = 200
        ge, gn, gu = ellipsoid_gravity(coordinates, ellipsoid, density, field="g")

        ellipsoid_volume = 4 / 3 * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c
        point_mass = density * ellipsoid_volume
        ge_point, gn_point, gz_point = tuple(
            point_gravity(coordinates, ellipsoid.centre, point_mass, field=f)
            for f in ("g_e", "g_n", "g_z")
        )

        rtol = 1e-5
        np.testing.assert_allclose(ge, ge_point, rtol=rtol)
        np.testing.assert_allclose(gn, gn_point, rtol=rtol)
        np.testing.assert_allclose(-gu, gz_point, rtol=rtol)

    def test_convergence(self, ellipsoid):
        """
        Test if ellipsoid gravity fields converges to the one of a point source.
        """
        phi, theta = 48.9, 12.3
        max_semiaxis = max((ellipsoid.a, ellipsoid.b, ellipsoid.c))
        radii = np.linspace(max_semiaxis * 1e3, max_semiaxis * 1e4, 51)
        coordinates = (
            radii * np.cos(phi) * np.cos(theta),
            radii * np.sin(phi) * np.cos(theta),
            radii * np.sin(theta),
        )
        density = 200
        ge, gn, gu = ellipsoid_gravity(coordinates, ellipsoid, density, field="g")

        ellipsoid_volume = 4 / 3 * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c
        point_mass = density * ellipsoid_volume
        ge_point, gn_point, gz_point = tuple(
            point_gravity(coordinates, ellipsoid.centre, point_mass, field=f)
            for f in ("g_e", "g_n", "g_z")
        )

        # Test if difference between fields gets smaller with distance
        ge_diff = np.abs(ge - ge_point)
        assert np.all(ge_diff[:-1] > ge_diff[1:])

        gn_diff = np.abs(gn - gn_point)
        assert np.all(gn_diff[:-1] > gn_diff[1:])

        gu_diff = np.abs(gu - -gz_point)
        assert np.all(gu_diff[:-1] > gu_diff[1:])
