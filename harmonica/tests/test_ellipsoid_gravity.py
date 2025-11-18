# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
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
from harmonica.errors import NoPhysicalPropertyWarning

from .._forward.ellipsoid_gravity import (
    ellipsoid_gravity,
)
from .._forward.ellipsoids import (
    OblateEllipsoid,
    ProlateEllipsoid,
    Sphere,
    TriaxialEllipsoid,
)


def build_ellipsoid(ellipsoid_type, *, center=(0, 0, 0), density=None):
    """
    Build a sample ellipsoid.

    Parameters
    ----------
    ellipsoid_type : {"triaxial", "prolate", "oblate", "sphere"}

    Returns
    -------
    ellipsoid
    """
    match ellipsoid_type:
        case "triaxial":
            a, b, c = 3.2, 2.1, 1.3
            ellipsoid = TriaxialEllipsoid(
                a=a, b=b, c=c, yaw=0, pitch=0, roll=0, center=center, density=density
            )
        case "prolate":
            a, b = 3.2, 2.1
            ellipsoid = ProlateEllipsoid(
                a=a, b=b, yaw=0, pitch=0, center=center, density=density
            )
        case "oblate":
            a, b = 2.2, 3.1
            ellipsoid = OblateEllipsoid(
                a=a, b=b, yaw=0, pitch=0, center=center, density=density
            )
        case "sphere":
            a = 3.2
            ellipsoid = Sphere(a=a, center=center, density=density)
        case _:
            msg = f"Invalid ellipsoid type: {ellipsoid_type}"
            raise ValueError(msg)
    return ellipsoid


def test_degenerate_ellipsoid_cases():
    """
    Test cases where the ellipsoid axes lengths are close to the boundary of
    accepted values.

    """
    # ellipsoids take (a, b, #c, yaw, pitch, #roll, center)
    a, b, c = 5, 4.99999999, 4.99999998
    yaw, pitch, roll = 0, 0, 0
    center = (0, 0, 0)
    density = 2000
    tri = TriaxialEllipsoid(a, b, c, yaw, pitch, roll, center, density=density)
    pro = ProlateEllipsoid(a, b, yaw, pitch, center, density=density)
    obl = OblateEllipsoid(b, a, yaw, pitch, center, density=density)
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    _, _, _ = ellipsoid_gravity(coordinates, tri)
    _, _, _ = ellipsoid_gravity(coordinates, pro)
    _, _, _ = ellipsoid_gravity(coordinates, obl)


def test_opposite_planes():
    """

    Test two surfaces produce the same anomaly but 'flipped' when including a
    rotation in the ellipsoid.

    """
    a, b, c = (4, 3, 2)  # triaxial ellipsoid
    yaw, pitch, roll = 90, 0, 0
    center = (0, 0, 0)
    density = 2000
    triaxial_example = TriaxialEllipsoid(
        a, b, c, yaw, pitch, roll, center, density=density
    )

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates1 = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )
    coordinates2 = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=-5
    )

    _, _, gz1 = ellipsoid_gravity(coordinates1, triaxial_example)
    _, _, gz2 = ellipsoid_gravity(coordinates2, triaxial_example)
    np.testing.assert_allclose(gz1, -np.flip(gz2))


def test_int_ext_boundary():
    """

    Check that the boundary of the internal and eternal components of the
    calculation align.

    """

    # compare a set value apart
    a, b, c = (5, 4, 3)
    ellipsoid = TriaxialEllipsoid(
        a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0), density=2000
    )

    e = np.array([[4.9999999, 5.00000001]])
    n = np.array([[0.0, 0.0]])
    u = np.array([[0.0, 0.0]])
    coordinates = (e, n, u)

    ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid)

    np.testing.assert_allclose(ge[0, 0], ge[0, 1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gn[0, 0], gn[0, 1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gz[0, 0], gz[0, 1], rtol=1e-5, atol=1e-5)


class TestSymmetry:
    """
    Test symmetry in gravity fields.
    """

    @pytest.fixture(params=["triaxial", "prolate", "oblate", "sphere"])
    def ellipsoid(self, request):
        """
        Sample ellipsoid.
        """
        ellipsoid_type = request.param
        return build_ellipsoid(ellipsoid_type, density=200)

    def test_vertical_symmetry_on_surface(self, ellipsoid):
        """
        Test symmetry of gz across a vertical axis that passes through the center of the
        ellipsoid.
        """
        points = [(0, 0, ellipsoid.c), (0, 0, -ellipsoid.c)]
        (_, _, gz_up), (_, _, gz_down) = tuple(
            ellipsoid_gravity(p, ellipsoid) for p in points
        )
        np.testing.assert_allclose(gz_up, -gz_down)

    @pytest.mark.parametrize("ellipsoid_type", ["oblate", "prolate"])
    @pytest.mark.parametrize("points", ["internal", "surface", "external"])
    def test_symmetry_on_circle(self, points, ellipsoid_type):
        """
        Test symmetry of |g| on circle around center of a prolate and oblate ellipsoid.

        Define a circle in the northing-upward plane, compute |g| on points along that
        circle. All values of |g| should be equal for prolate and oblate ellipsoids.
        """
        ellipsoid = build_ellipsoid(ellipsoid_type, density=200)

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
        ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid)
        g = np.sqrt(ge**2 + gn**2 + gz**2)

        # Check that |g| is constant in the circle
        np.testing.assert_allclose(g[0], g)

    @pytest.mark.parametrize("points", ["internal", "surface", "external"])
    def test_symmetry_on_sphere(self, points):
        """
        Test symmetry of |g| on sphere around center of a sphere.

        Define a sphere around the sphere, compute |g| on the sphere.
        All values of |g| should be equal for spherical ellipsoids.
        """
        sphere = build_ellipsoid("sphere", density=200)

        # Build coordinates along sphere centered in the center of the ellipsoid
        phi = np.linspace(0, 2 * np.pi, 61)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 19)
        phi, theta = np.meshgrid(phi, theta)

        match points:
            case "surface":
                radius = sphere.a
            case "internal":
                radius = sphere.a * 0.5
            case "external":
                radius = sphere.a * 1.5
            case _:
                raise ValueError()

        easting = radius * np.cos(phi) * np.cos(theta)
        northing = radius * np.sin(phi) * np.cos(theta)
        upward = radius * np.sin(theta)
        coordinates = (easting, northing, upward)

        # Compute gravity acceleration
        ge, gn, gz = ellipsoid_gravity(coordinates, sphere)
        g = np.sqrt(ge**2 + gn**2 + gz**2).ravel()

        # Check that |g| is constant in the sphere
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
        return build_ellipsoid(ellipsoid_type, density=200)

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
        ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid)

        ellipsoid_volume = 4 / 3 * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c
        point_mass = ellipsoid.density * ellipsoid_volume
        ge_point, gn_point, gz_point = tuple(
            point_gravity(coordinates, ellipsoid.center, point_mass, field=f)
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
        ge, gn, gz = ellipsoid_gravity(coordinates, ellipsoid)

        ellipsoid_volume = 4 / 3 * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c
        point_mass = ellipsoid.density * ellipsoid_volume
        ge_point, gn_point, gz_point = tuple(
            point_gravity(coordinates, ellipsoid.center, point_mass, field=f)
            for f in ("g_e", "g_n", "g_z")
        )

        # Test if difference between fields gets smaller with distance
        ge_diff = np.abs(ge - ge_point)
        assert np.all(ge_diff[:-1] > ge_diff[1:])

        gn_diff = np.abs(gn - gn_point)
        assert np.all(gn_diff[:-1] > gn_diff[1:])

        gz_diff = np.abs(gz - gz_point)
        assert np.all(gz_diff[:-1] > gz_diff[1:])


class TestSphereVsPointSource:
    """
    Test if gravity field of sphere is equal to the one of a point source.

    For any external point, the gravity field of the sphere should be the same as the
    one of a point source located in the center of the sphere.
    """

    def test_sphere_vs_point_source(self):
        """
        Compare gravity acceleration of sphere on external points with the point source.
        """
        # Define sphere
        radius = 50.0
        center = (10, -29, 105)
        density = 200.0
        sphere = Sphere(radius, center=center, density=density)

        # Build a 3d grid of observation points centered in (0, 0, 0)
        n = 51
        extent = 3 * radius
        easting, northing, upward = np.meshgrid(
            *[np.linspace(-extent, extent, n) for _ in range(3)]
        )

        # Remove the observation points that lie inside the sphere
        inside = easting**2 + northing**2 + upward**2 < radius**2
        easting = easting[~inside]
        northing = northing[~inside]
        upward = upward[~inside]

        # Shift the coordinates, so they are centered around the sphere
        coordinates = (easting + center[0], northing + center[1], upward + center[2])

        # Forward model the sphere
        g_sphere = ellipsoid_gravity(coordinates, sphere)

        # Forward model a point source located in the center of the sphere
        volume = 4 / 3 * np.pi * radius**3
        g_point = tuple(
            point_gravity(coordinates, center, masses=volume * density, field=field)
            for field in ("g_e", "g_n", "g_z")
        )

        np.testing.assert_allclose(g_sphere, g_point)


class TestEllipsoidVsSphere:
    """
    Compare gravity field of ellipsoids with the one of a sphere.

    If the ellipsoids semiaxes are close enough to each other, their fields should be
    close enough to the ones of a sphere, both inside and outside the bodies.
    """

    # Sphere radius, center, and susceptibility.
    radius = 50.0
    center = (28, 19, -50)
    density = 200.0

    # Difference between ellipsoid's semiaxes.
    # It should be small compared to the sphere radius, so the ellipsoid approximates
    # a sphere.
    delta = 1e-4

    @pytest.fixture
    def sphere(self):
        """Sphere used to compare gravity fields."""
        return Sphere(self.radius, self.center, density=self.density)

    @pytest.fixture(params=["oblate", "prolate", "triaxial"])
    def ellipsoid(self, request):
        """
        Ellipsoid that approximates a sphere.
        """
        yaw, pitch, roll = 0, 0, 0
        a = self.radius
        match request.param:
            case "oblate":
                ellipsoid = OblateEllipsoid(
                    a=a,
                    b=a + self.delta,
                    yaw=yaw,
                    pitch=pitch,
                    center=self.center,
                    density=self.density,
                )
            case "prolate":
                ellipsoid = ProlateEllipsoid(
                    a=a,
                    b=a - self.delta,
                    yaw=yaw,
                    pitch=pitch,
                    center=self.center,
                    density=self.density,
                )
            case "triaxial":
                ellipsoid = TriaxialEllipsoid(
                    a=a,
                    b=a - self.delta,
                    c=a - 2 * self.delta,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    center=self.center,
                    density=self.density,
                )
            case _:
                raise ValueError()
        return ellipsoid

    @pytest.fixture
    def coordinates(self):
        """3D grid of observation points around the sphere."""
        n, extent = 51, 2 * self.radius
        easting, northing, upward = np.meshgrid(
            *[np.linspace(-extent, extent, n) for _ in range(3)]
        )
        coordinates = (
            easting + self.center[0],
            northing + self.center[1],
            upward + self.center[2],
        )
        return coordinates

    def test_ellipsoid_vs_sphere(self, coordinates, ellipsoid, sphere):
        """
        Compare gravity field of ellipsoids against the one for a sphere.
        """
        # Forward model the gravity acceleration for sphere and ellipsoid
        g_sphere = ellipsoid_gravity(coordinates, sphere)
        g_ellipsoid = ellipsoid_gravity(coordinates, ellipsoid)

        # Compare the two fields
        maxabs = vd.maxabs(*g_sphere, *g_ellipsoid)
        atol = maxabs * 1e-5
        np.testing.assert_allclose(g_sphere, g_ellipsoid, atol=atol)


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
        density = 238
        match ellipsoid_type:
            case "oblate":
                ellipsoid = OblateEllipsoid(
                    a=semiminor,
                    b=semimajor,
                    yaw=yaw,
                    pitch=pitch,
                    center=center,
                    density=density,
                )
            case "prolate":
                ellipsoid = ProlateEllipsoid(
                    a=semimajor,
                    b=semiminor,
                    yaw=yaw,
                    pitch=pitch,
                    center=center,
                    density=density,
                )
            case "triaxial":
                ellipsoid = TriaxialEllipsoid(
                    a=semimajor,
                    b=semimiddle,
                    c=semiminor,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    center=center,
                    density=density,
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
        g_field, g_field_flipped = tuple(
            ellipsoid_gravity(coordinates, ell)
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
                a=20,
                b=60,
                yaw=30.2,
                pitch=-23,
                center=(-10.0, 20.0, -10.0),
                density=200.0,
            ),
            ProlateEllipsoid(
                a=40,
                b=15,
                yaw=170.2,
                pitch=71,
                center=(15.0, 0.0, -40.0),
                density=-400.0,
            ),
            TriaxialEllipsoid(
                a=60,
                b=18,
                c=15,
                yaw=272.1,
                pitch=43,
                roll=98,
                center=(0.0, 20.0, -30.0),
                density=700.0,
            ),
        ]
        return ellipsoids

    def test_multiple_ellipsoids(self, coordinates, ellipsoids):
        # Compute gravity acceleration
        gx, gy, gz = ellipsoid_gravity(coordinates, ellipsoids)

        # Compute expected arrays
        gx_expected, gy_expected, gz_expected = tuple(
            np.zeros_like(coordinates[0]) for _ in range(3)
        )
        for ellipsoid in ellipsoids:
            gx_i, gy_i, gz_i = ellipsoid_gravity(coordinates, ellipsoid)
            gx_expected += gx_i
            gy_expected += gy_i
            gz_expected += gz_i

        # Check if fields are the same
        np.testing.assert_allclose(gx, gx_expected)
        np.testing.assert_allclose(gy, gy_expected)
        np.testing.assert_allclose(gz, gz_expected)


@pytest.mark.parametrize(
    "ellipsoid_class", [OblateEllipsoid, ProlateEllipsoid, TriaxialEllipsoid]
)
class TestNoneDensity:
    """Test warning when ellipsoid has no density."""

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
        elif ellipsoid_class is Sphere:
            args = {"a": 50.0, "center": (0, 0, 0)}
        else:
            raise TypeError()
        return args

    def test_warning(self, ellipsoid_class, ellipsoid_args):
        """
        Test warning about ellipsoid with no density being skipped.
        """
        coordinates = (0.0, 0.0, 0.0)
        ellipsoid = ellipsoid_class(**ellipsoid_args)

        msg = re.escape(
            f"Ellipsoid {ellipsoid} doesn't have a density value. It will be skipped."
        )
        with pytest.warns(NoPhysicalPropertyWarning, match=msg):
            gx, gy, gz = ellipsoid_gravity(coordinates, ellipsoid)

        # Check the gravity acceleration components are zero
        for g_component in (gx, gy, gz):
            assert g_component == 0.0
