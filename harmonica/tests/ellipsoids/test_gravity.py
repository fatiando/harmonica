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

from harmonica import ellipsoid_gravity, point_gravity
from harmonica._forward.ellipsoids import Ellipsoid
from harmonica._forward.ellipsoids.utils import SEMIAXES_RTOL
from harmonica.errors import NoPhysicalPropertyWarning


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
        case "prolate":
            a, b = 3.2, 2.1
            c = b
        case "oblate":
            a, b = 2.2, 3.1
            c = b
        case "sphere":
            a = 3.2
            b, c = a, a
        case _:
            msg = f"Invalid ellipsoid type: {ellipsoid_type}"
            raise ValueError(msg)

    ellipsoid = Ellipsoid(a, b, c, center=center, density=density)
    return ellipsoid


def test_opposite_planes():
    """

    Test two surfaces produce the same anomaly but 'flipped' when including a
    rotation in the ellipsoid.

    """
    a, b, c = (4, 3, 2)  # triaxial ellipsoid
    density = 2000
    triaxial_example = Ellipsoid(a, b, c, density=density)

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
    ellipsoid = Ellipsoid(a, b, c, density=2000)

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
        sphere = Ellipsoid(radius, radius, radius, center=center, density=density)

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

    # Ratio between ellipsoid's semiaxes, defined as ratio = | a - b | /  max(a, b).
    # Make it small enough so ellipsoids approximate a sphere, but not too small that
    # might trigger the fixes for numerical uncertainties.
    ratio = 1e-4
    assert ratio > SEMIAXES_RTOL

    @pytest.fixture
    def sphere(self):
        """Sphere used to compare gravity fields."""
        return Ellipsoid(
            self.radius,
            self.radius,
            self.radius,
            center=self.center,
            density=self.density,
        )

    @pytest.fixture(params=["oblate", "prolate", "triaxial"])
    def ellipsoid(self, request):
        """
        Ellipsoid that approximates a sphere.
        """
        a = self.radius
        match request.param:
            case "oblate":
                a = b = self.radius
                c = (1 - self.ratio) * b
            case "prolate":
                a = self.radius
                b = c = (1 - self.ratio) * a
            case "triaxial":
                a = self.radius
                b = (1 - self.ratio) * a
                c = (1 - 2 * self.ratio) * a
            case _:
                raise ValueError()
        ellipsoid = Ellipsoid(a, b, c, center=self.center, density=self.density)
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
        np.testing.assert_allclose(g_sphere, g_ellipsoid, rtol=7e-4)


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
                a, b = semiminor, semimajor
                c = b
            case "prolate":
                a, b = semimajor, semiminor
                c = b
            case "triaxial":
                a, b, c = semimajor, semimiddle, semiminor
                c = b
            case _:
                raise ValueError()
        ellipsoid = Ellipsoid(
            a, b, c, yaw=yaw, pitch=pitch, roll=roll, center=center, density=density
        )
        return ellipsoid

    def test_symmetry_when_flipping(self, ellipsoid):
        """
        Test symmetry of gravity field when flipping the ellipsoid.

        Rotate the ellipsoid so the geometry is preserved. The gravity field generated
        by the ellipsoid should be the same as before the rotation.
        """
        # Define observation points
        coordinates = vd.grid_coordinates(
            region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
        )

        # Generate a flipped ellipsoid
        ellipsoid_flipped = self.flip_ellipsoid(copy(ellipsoid))

        # Compute gravity fields
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
            Ellipsoid(
                a=20,
                b=60,
                c=60,
                yaw=30.2,
                pitch=-23,
                center=(-10.0, 20.0, -10.0),
                density=200.0,
            ),
            Ellipsoid(
                a=40,
                b=15,
                c=15,
                yaw=170.2,
                pitch=71,
                center=(15.0, 0.0, -40.0),
                density=-400.0,
            ),
            Ellipsoid(
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


class TestNoneDensity:
    """Test warning when ellipsoid has no density."""

    def test_warning(self):
        """
        Test warning about ellipsoid with no density being skipped.
        """
        coordinates = (0.0, 0.0, 0.0)
        ellipsoid = build_ellipsoid("triaxial")

        msg = re.escape(
            f"Ellipsoid {ellipsoid} doesn't have a density value. It will be skipped."
        )
        with pytest.warns(NoPhysicalPropertyWarning, match=msg):
            gx, gy, gz = ellipsoid_gravity(coordinates, ellipsoid)

        # Check the gravity acceleration components are zero
        for g_component in (gx, gy, gz):
            assert g_component == 0.0


class TestNumericalInstability:
    """
    Test fix for numerical instabilities when semiaxes lengths are very similar.

    When the semiaxes are almost equal to each other (in the order of machine
    precision), analytic solutions for prolate, oblate, and triaxial ellipsoids might
    fall under singularities, triggering numerical instabilities.

    Given two semiaxes a and b, define a ratio as:

    .. code::

        ratio  = | a - b | / max(a, b)

    These test functions will build the ellipsoids using very small values of this
    ratio.
    """

    # Properties of the sphere
    radius = 50.0
    center = (0, 0, 0)
    density = 200.0

    def get_coordinates(self, azimuth=45, polar=45):
        """
        Generate coordinates of observation points along a certain direction.
        """
        r = np.linspace(self.radius, 5 * self.radius, 501)
        azimuth, polar = np.rad2deg(azimuth), np.rad2deg(polar)
        easting = r * np.cos(azimuth) * np.cos(polar)
        northing = r * np.sin(azimuth) * np.cos(polar)
        upward = r * np.sin(polar)
        return (easting, northing, upward)

    @pytest.fixture
    def sphere(self):
        a = self.radius
        return Ellipsoid(a, a, a, center=self.center, density=self.density)

    def test_gravity_prolate(self, sphere):
        """
        Test gravity field of prolate ellipsoid that is almost a sphere.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-9

        a = self.radius
        b = (1 - ratio) * a
        ellipsoid = Ellipsoid(a, b, b, center=self.center, density=self.density)
        ge_sphere, gn_sphere, gu_sphere = ellipsoid_gravity(coordinates, sphere)
        ge_ell, gn_ell, gu_ell = ellipsoid_gravity(coordinates, ellipsoid)

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(ge_ell, ge_sphere, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gn_ell, gn_sphere, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gu_ell, gu_sphere, rtol=rtol, atol=atol)

    def test_gravity_oblate(self, sphere):
        """
        Test gravity field of oblate ellipsoid that is almost a sphere.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-14

        a = self.radius
        b = (1 + ratio) * a
        ellipsoid = Ellipsoid(a, a, b, center=self.center, density=self.density)
        ge_sphere, gn_sphere, gu_sphere = ellipsoid_gravity(coordinates, sphere)
        ge_ell, gn_ell, gu_ell = ellipsoid_gravity(coordinates, ellipsoid)

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(ge_ell, ge_sphere, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gn_ell, gn_sphere, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gu_ell, gu_sphere, rtol=rtol, atol=atol)

    def test_gravity_triaxial(self, sphere):
        """
        Test gravity field of triaxial ellipsoid that is almost a sphere.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-14

        a = self.radius
        b = (1 - ratio) * a
        c = (1 - 2 * ratio) * a
        ellipsoid = Ellipsoid(a, b, c, center=self.center, density=self.density)
        ge_sphere, gn_sphere, gu_sphere = ellipsoid_gravity(coordinates, sphere)
        ge_ell, gn_ell, gu_ell = ellipsoid_gravity(coordinates, ellipsoid)

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(ge_ell, ge_sphere, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gn_ell, gn_sphere, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gu_ell, gu_sphere, rtol=rtol, atol=atol)


class TestTriaxialOnLimits:
    """
    Test a triaxial ellipsoid vs oblate and prolate ellipsoids.

    Test the gravity fields of a triaxial ellipsoid that approximates an oblate and
    a prolate one against their gravity fields.
    """

    semimajor = 50.0
    atol_ratio = 1e-5
    rtol = 1e-5

    def get_coordinates(self, azimuth=45, polar=45):
        """
        Generate coordinates of observation points along a certain direction.
        """
        r = np.linspace(self.semimajor, 5 * self.semimajor, 501)
        azimuth, polar = np.rad2deg(azimuth), np.rad2deg(polar)
        easting = r * np.cos(azimuth) * np.cos(polar)
        northing = r * np.sin(azimuth) * np.cos(polar)
        upward = r * np.sin(polar)
        return (easting, northing, upward)

    def test_triaxial_vs_prolate(self):
        """Compare triaxial with prolate ellipsoid."""
        coordinates = self.get_coordinates()
        a, b = self.semimajor, 20.0
        c = b - 1e-4
        density = 200.0
        triaxial = Ellipsoid(a, b, c, density=density)
        prolate = Ellipsoid(a, b, b, density=density)

        g_triaxial, g_prolate = tuple(
            ellipsoid_gravity(coordinates, ell) for ell in (triaxial, prolate)
        )

        for gi_triaxial, gi_prolate in zip(g_triaxial, g_prolate, strict=True):
            atol = self.atol_ratio * vd.maxabs(gi_prolate)
            np.testing.assert_allclose(
                gi_triaxial, gi_prolate, atol=atol, rtol=self.rtol
            )

    def test_triaxial_vs_oblate(self):
        """Compare triaxial with oblate ellipsoid."""
        coordinates = self.get_coordinates()
        a = self.semimajor
        b = a - 1e-4
        c = 20.0
        density = 200.0
        triaxial = Ellipsoid(a, b, c, density=density)
        oblate = Ellipsoid(a, a, c, density=density)

        g_triaxial, g_oblate = tuple(
            ellipsoid_gravity(coordinates, ell) for ell in (triaxial, oblate)
        )

        for gi_triaxial, gi_oblate in zip(g_triaxial, g_oblate, strict=True):
            atol = self.atol_ratio * vd.maxabs(gi_oblate)
            np.testing.assert_allclose(
                gi_triaxial, gi_oblate, atol=atol, rtol=self.rtol
            )


class TestSemiaxesArbitraryOrder:
    """
    Test gravity fields when defining the same ellipsoids with different orders of the
    semiaxes plus needed rotation angles.
    """

    atol_ratio = 0.0
    rtol = 1e-7

    def get_coordinates(self, semimajor, azimuth=45, polar=45):
        """
        Generate coordinates of observation points along a certain direction.
        """
        r = np.linspace(0.5 * semimajor, 5.5 * semimajor, 501)
        azimuth, polar = np.rad2deg(azimuth), np.rad2deg(polar)
        easting = r * np.cos(azimuth) * np.cos(polar)
        northing = r * np.sin(azimuth) * np.cos(polar)
        upward = r * np.sin(polar)
        return (easting, northing, upward)

    @pytest.mark.parametrize(
        ("semiaxes", "yaw", "pitch", "roll"),
        [
            ((30, 20, 10), 0, 0, 0),
            ((20, 30, 10), 90, 0, 0),
            ((10, 20, 30), 0, 90, 0),
            ((30, 10, 20), 0, 0, 90),
            ((20, 10, 30), 90, 90, 0),
            ((10, 30, 20), 90, 0, 90),
        ],
    )
    def test_triaxial(self, semiaxes, yaw, pitch, roll):
        a, b, c = semiaxes
        semiaxes_sorted = sorted(semiaxes, reverse=True)
        density = 200.0
        ellipsoid = Ellipsoid(a, b, c, density=density)
        ellipsoid_rotated = Ellipsoid(
            *semiaxes_sorted, yaw=yaw, pitch=pitch, roll=roll, density=density
        )

        coordinates = self.get_coordinates(semiaxes_sorted[0])
        g_field = ellipsoid_gravity(coordinates, ellipsoid)
        g_rotated = ellipsoid_gravity(coordinates, ellipsoid_rotated)

        np.testing.assert_allclose(g_field, g_rotated, rtol=self.rtol)

    @pytest.mark.parametrize(
        ("semiaxes", "yaw", "pitch", "roll"),
        [
            ((30, 10, 10), 0, 0, 0),
            ((10, 30, 10), 90, 0, 0),
            ((10, 10, 30), 0, 90, 0),
        ],
    )
    def test_prolate(self, semiaxes, yaw, pitch, roll):
        a, b, c = semiaxes
        semiaxes_sorted = sorted(semiaxes, reverse=True)
        density = 200.0
        ellipsoid = Ellipsoid(a, b, c, density=density)
        ellipsoid_rotated = Ellipsoid(
            *semiaxes_sorted, yaw=yaw, pitch=pitch, roll=roll, density=density
        )

        coordinates = self.get_coordinates(semiaxes_sorted[0])
        g_field = ellipsoid_gravity(coordinates, ellipsoid)
        g_rotated = ellipsoid_gravity(coordinates, ellipsoid_rotated)

        np.testing.assert_allclose(g_field, g_rotated, rtol=self.rtol)

    @pytest.mark.parametrize(
        ("semiaxes", "yaw", "pitch", "roll"),
        [
            ((20, 20, 10), 0, 0, 0),
            ((20, 10, 20), 0, 0, 90),
            ((10, 20, 20), 0, 90, 0),
        ],
    )
    def test_oblate(self, semiaxes, yaw, pitch, roll):
        a, b, c = semiaxes
        semiaxes_sorted = sorted(semiaxes, reverse=True)
        density = 200.0
        ellipsoid = Ellipsoid(a, b, c, density=density)
        ellipsoid_rotated = Ellipsoid(
            *semiaxes_sorted, yaw=yaw, pitch=pitch, roll=roll, density=density
        )

        coordinates = self.get_coordinates(semiaxes_sorted[0])
        g_field = ellipsoid_gravity(coordinates, ellipsoid)
        g_rotated = ellipsoid_gravity(coordinates, ellipsoid_rotated)

        np.testing.assert_allclose(g_field, g_rotated, rtol=self.rtol)


class TestNumericalInstabilitiesTriaxial:
    """
    Test fix for numerical instabilities when triaxial approximates a prolate or oblate.

    When two of the three semiaxes of a triaxial ellipsoid are almost equal to each
    other (in the order of machine precision), analytic solutions might fall under
    singularities, triggering numerical instabilities.

    Given two semiaxes a and b, define a ratio as:

    .. code::

        ratio  = | a - b | / max(a, b)

    These test functions will build the ellipsoids using very small values of this
    ratio.
    """

    # Properties of the sphere
    semimajor, semiminor = 50.0, 30.0
    center = (0, 0, 0)
    density = 200.0

    def get_coordinates(self, azimuth=45, polar=45):
        """
        Generate coordinates of observation points along a certain direction.
        """
        r = np.linspace(0.5 * self.semimajor, 5.5 * self.semimajor, 501)
        azimuth, polar = np.rad2deg(azimuth), np.rad2deg(polar)
        easting = r * np.cos(azimuth) * np.cos(polar)
        northing = r * np.sin(azimuth) * np.cos(polar)
        upward = r * np.sin(polar)
        return (easting, northing, upward)

    @pytest.fixture
    def prolate(self):
        return Ellipsoid(
            self.semimajor,
            self.semiminor,
            self.semiminor,
            center=self.center,
            density=self.density,
        )

    @pytest.fixture
    def oblate(self):
        return Ellipsoid(
            self.semimajor,
            self.semimajor,
            self.semiminor,
            center=self.center,
            density=self.density,
        )

    def test_triaxial_vs_prolate(self, prolate):
        """
        Test gravity field of triaxial ellipsoid that is almost a prolate.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-14

        a = self.semimajor
        b = self.semiminor
        c = (1 - ratio) * b
        ellipsoid = Ellipsoid(a, b, c, center=self.center, density=self.density)
        ge_prolate, gn_prolate, gz_prolate = ellipsoid_gravity(coordinates, prolate)
        ge_ell, gn_ell, gz_ell = ellipsoid_gravity(coordinates, ellipsoid)

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(ge_ell, ge_prolate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gn_ell, gn_prolate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gz_ell, gz_prolate, rtol=rtol, atol=atol)

    def test_triaxial_vs_oblate(self, oblate):
        """
        Test gravity field of triaxial ellipsoid that is almost a oblate.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-14

        a = self.semimajor
        b = (1 - ratio) * a
        c = self.semiminor
        ellipsoid = Ellipsoid(a, b, c, center=self.center, density=self.density)
        ge_oblate, gn_oblate, gz_oblate = ellipsoid_gravity(coordinates, oblate)
        ge_ell, gn_ell, gz_ell = ellipsoid_gravity(coordinates, ellipsoid)

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(ge_ell, ge_oblate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gn_ell, gn_oblate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gz_ell, gz_oblate, rtol=rtol, atol=atol)
