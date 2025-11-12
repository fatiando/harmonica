"""
Test gravity forward modelling of ellipsoids.
"""

# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import re
from copy import copy

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


def build_ellipsoid(ellipsoid_type):
    """
    Build a sample ellipsoid.

    Parameters
    ----------
    ellipsoid_type : {"triaxial", "prolate", "oblate"}

    Returns
    -------
    ellipsoid
    """
    centre = (0, 0, 0)
    match ellipsoid_type:
        case "triaxial":
            a, b, c = 3.2, 2.1, 1.3
            ellipsoid = TriaxialEllipsoid(
                a=a, b=b, c=c, yaw=0, pitch=0, roll=0, centre=centre
            )
        case "prolate":
            a, b = 3.2, 2.1
            ellipsoid = ProlateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
        case "oblate":
            a, b = 2.2, 3.1
            ellipsoid = OblateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
        case _:
            msg = f"Invalid ellipsoid type: {ellipsoid_type}"
            raise ValueError(msg)
    return ellipsoid


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

    _, _, _ = ellipsoid_gravity(coordinates, tri, density, field="g")
    _, _, _ = ellipsoid_gravity(coordinates, pro, density, field="g")
    _, _, _ = ellipsoid_gravity(coordinates, obl, density, field="g")


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

    _, _, gz1 = ellipsoid_gravity(coordinates1, triaxial_example, density, field="g")
    _, _, gz2 = ellipsoid_gravity(coordinates2, triaxial_example, density, field="g")
    np.testing.assert_allclose(gz1, -np.flip(gz2))


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

    ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid, 2000, field="g")

    np.testing.assert_allclose(ge[0, 0], ge[0, 1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gn[0, 0], gn[0, 1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gz[0, 0], gz[0, 1], rtol=1e-5, atol=1e-5)


class TestSymmetry:
    """
    Test symmetry in gravity fields.
    """

    @pytest.fixture(params=["triaxial", "prolate", "oblate"])
    def ellipsoid(self, request):
        """
        Sample ellipsoid.
        """
        ellipsoid_type = request.param
        return build_ellipsoid(ellipsoid_type)

    def test_vertical_symmetry_on_surface(self, ellipsoid):
        """
        Test symmetry of gz across a vertical axis that passes through the center of the
        ellipsoid.
        """
        points = [(0, 0, ellipsoid.c), (0, 0, -ellipsoid.c)]
        density = 200
        gz_up, gz_down = tuple(
            ellipsoid_gravity(p, ellipsoid, density, field="g_z") for p in points
        )
        np.testing.assert_allclose(gz_up, -gz_down)

    @pytest.mark.parametrize("ellipsoid_type", ["oblate", "prolate"])
    @pytest.mark.parametrize("points", ["internal", "surface", "external"])
    def test_symmetry_on_circle(self, points, ellipsoid_type):
        """
        Test symmetry of |g| on circle around center of a prolate and oblate ellipsoid.

        Define a circle in the northing-upward plane, compute |g| on points along that
        circle. All values of |g| should be equal.
        """
        ellipsoid = build_ellipsoid(ellipsoid_type)

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
        ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid, density, field="g")
        g = np.sqrt(ge**2 + gn**2 + gz**2)

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
        match ellipsoid_type:
            case "triaxial":
                a, b, c = 3.2, 2.1, 1.3
                ellipsoid = TriaxialEllipsoid(
                    a=a, b=b, c=c, yaw=0, pitch=0, roll=0, centre=centre
                )
            case "prolate":
                a, b = 3.2, 2.1
                ellipsoid = ProlateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
            case "oblate":
                a, b = 2.2, 3.1
                ellipsoid = OblateEllipsoid(a=a, b=b, yaw=0, pitch=0, centre=centre)
            case _:
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
        ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid, density, field="g")

        ellipsoid_volume = 4 / 3 * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c
        point_mass = density * ellipsoid_volume
        ge_point, gn_point, gz_point = tuple(
            point_gravity(coordinates, ellipsoid.centre, point_mass, field=f)
            for f in ("g_e", "g_n", "g_z")
        )

        rtol = 1e-5
        np.testing.assert_allclose(ge, ge_point, rtol=rtol)
        np.testing.assert_allclose(gn, gn_point, rtol=rtol)
        np.testing.assert_allclose(gz, gz_point, rtol=rtol)

    def test_convergence(self, ellipsoid):
        """
        Test if ellipsoid gravity fields converges to the one of a point source.
        """
        phi, theta = 48.9, 12.3
        max_semiaxis = max((ellipsoid.a, ellipsoid.b, ellipsoid.c))
        radii = np.linspace(max_semiaxis * 10, max_semiaxis * 400, 51)
        coordinates = (
            radii * np.cos(phi) * np.cos(theta),
            radii * np.sin(phi) * np.cos(theta),
            radii * np.sin(theta),
        )
        density = 200
        ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid, density, field="g")

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

        gz_diff = np.abs(gz - gz_point)
        assert np.all(gz_diff[:-1] > gz_diff[1:])


def test_invalid_field():
    """
    Test error after invalid field.
    """
    ellipsoid = build_ellipsoid("prolate")
    coordinates = (1, 2, 3)
    invalid_field = "blah"
    msg = re.escape(f"Invalid field '{invalid_field}'")
    with pytest.raises(ValueError, match=msg):
        ellipsoid_gravity(coordinates, ellipsoid, density=1.0, field=invalid_field)


class TestSymmetryOnRotations:
    """
    Test symmetries in the gravity field after rotations are applied.
    """

    def flip_ellipsoid(self, ellipsoid):
        """
        Flip ellipsoid 180 degrees keeping the same geometry.

        The rotation will make the ellipsoid to turn 180 degrees, so its geometry is
        preserved. The sign change in pitch and roll is required to ensure the symmetry.
        """
        ellipsoid.yaw += 180
        ellipsoid.pitch *= -1
        if isinstance(ellipsoid, TriaxialEllipsoid):
            ellipsoid.roll *= -1
        return ellipsoid

    @pytest.fixture(params=["oblate", "prolate", "triaxial"])
    def ellipsoid(self, request):
        """Sample ellipsoid."""
        ellipsoid_type = request.param
        # Generate original ellipsoid
        semimajor, semimiddle, semiminor = 57.2, 42.0, 21.2
        center = (0, 0, 0)
        yaw, pitch, roll = 62.3, 48.2, 14.9
        match ellipsoid_type:
            case "oblate":
                ellipsoid = OblateEllipsoid(
                    a=semiminor, b=semimajor, yaw=yaw, pitch=pitch, centre=center
                )
            case "prolate":
                ellipsoid = ProlateEllipsoid(
                    a=semimajor, b=semiminor, yaw=yaw, pitch=pitch, centre=center
                )
            case "triaxial":
                ellipsoid = TriaxialEllipsoid(
                    a=semimajor,
                    b=semimiddle,
                    c=semiminor,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    centre=center,
                )
            case _:
                raise ValueError()
        return ellipsoid

    def test_symmetry_when_flipping(self, ellipsoid):
        """
        Test symmetry of magnetic field when flipping the ellipsoid.

        Rotate the ellipsoid so the geometry is preserved. The gravity field generated
        by the ellipsoid should be the same as before the rotation.
        """
        # Define observation points
        coordinates = vd.grid_coordinates(
            region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
        )

        # Generate a flipped ellipsoid
        ellipsoid_flipped = self.flip_ellipsoid(copy(ellipsoid))

        # Compute magnetic fields
        density = 238
        g_field, g_field_flipped = tuple(
            ellipsoid_gravity(
                coordinates,
                ell,
                density,
            )
            for ell in (ellipsoid, ellipsoid_flipped)
        )

        # Check that the gravity field is the same for original and flipped ellipsoids
        for i in range(3):
            np.testing.assert_allclose(g_field[i], g_field_flipped[i])


class TestMultipleEllipsoids:
    """
    Test forward function when passing multiple ellipsoids.
    """

    @pytest.fixture
    def coordinates(self):
        """Sample grid coordinates."""
        region = (-30, 30, -30, 30)
        coordinates = vd.grid_coordinates(
            region=region, shape=(21, 21), extra_coords=10
        )
        return coordinates

    @pytest.fixture
    def ellipsoids(self):
        """Sample ellipsoids."""
        ellipsoids = [
            OblateEllipsoid(
                a=20, b=60, yaw=30.2, pitch=-23, centre=(-10.0, 20.0, -10.0)
            ),
            ProlateEllipsoid(
                a=40, b=15, yaw=170.2, pitch=71, centre=(15.0, 0.0, -40.0)
            ),
            TriaxialEllipsoid(
                a=60,
                b=18,
                c=15,
                yaw=272.1,
                pitch=43,
                roll=98,
                centre=(0.0, 20.0, -30.0),
            ),
        ]
        return ellipsoids

    @pytest.fixture(params=["list", "array"])
    def densities(self, request):
        """Sample densities."""
        densities = [200.0, -400.0, 700.0]
        if request.param == "array":
            densities = np.array(densities)
        return densities

    def test_multiple_ellipsoids(self, coordinates, ellipsoids, densities):
        # Compute gravity acceleration
        gx, gy, gz = ellipsoid_gravity(coordinates, ellipsoids, densities)

        # Compute expected arrays
        gx_expected, gy_expected, gz_expected = tuple(
            np.zeros_like(coordinates[0]) for _ in range(3)
        )
        for ellipsoid, density in zip(ellipsoids, densities, strict=True):
            gx_i, gy_i, gz_i = ellipsoid_gravity(coordinates, ellipsoid, density)
            gx_expected += gx_i
            gy_expected += gy_i
            gz_expected += gz_i

        # Check if fields are the same
        np.testing.assert_allclose(gx, gx_expected)
        np.testing.assert_allclose(gy, gy_expected)
        np.testing.assert_allclose(gz, gz_expected)
