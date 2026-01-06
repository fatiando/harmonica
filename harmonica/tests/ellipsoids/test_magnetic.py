# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test magnetic forward modelling of ellipsoids.
"""

# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import itertools
import re
from copy import copy

import numpy as np
import pytest
import verde as vd
from scipy.constants import mu_0

import harmonica as hm
from harmonica import ellipsoid_magnetic
from harmonica._forward.ellipsoids import Ellipsoid
from harmonica._forward.ellipsoids.magnetic import (
    get_demagnetization_tensor_internal,
    get_magnetisation,
)
from harmonica._forward.ellipsoids.utils import SEMIAXES_RTOL
from harmonica._forward.utils import get_rotation_matrix
from harmonica.errors import NoPhysicalPropertyWarning


def test_euler_returns():
    """Check the euler returns are exact"""
    r0 = get_rotation_matrix(0, 0, 0)
    r360 = get_rotation_matrix(360, 0, 0)
    assert np.allclose(r0, r360)


def test_magnetic_symmetry():
    """
    Check the symmetry of magnetic calculations at surfaces above and below
    the body.
    """
    susceptibility = 0.1
    ellipsoid = Ellipsoid(4, 3, 2, susceptibility=susceptibility)

    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )
    coordinates2 = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=-5
    )

    external_field = (10_000, 0, 0)
    be1, bn1, bu1 = ellipsoid_magnetic(coordinates, ellipsoid, external_field)
    be2, bn2, bu2 = ellipsoid_magnetic(coordinates2, ellipsoid, external_field)

    np.testing.assert_allclose(np.abs(be1), np.flip(np.abs(be2)))
    np.testing.assert_allclose(np.abs(bn1), np.flip(np.abs(bn2)))
    np.testing.assert_allclose(np.abs(bu1), np.flip(np.abs(bu2)))


def test_flipped_h0():
    """
    Check that reversing the magentising field produces the same (reversed)
    field.
    """
    a, c = (2, 4)
    susceptibility = 0.1
    oblate = Ellipsoid(a, a, c, susceptibility=susceptibility)

    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    external_field1 = np.array((55_000, 0.0, 90.0))
    external_field2 = -external_field1
    be1, bn1, bu1 = ellipsoid_magnetic(coordinates, oblate, external_field1)
    be2, bn2, bu2 = ellipsoid_magnetic(coordinates, oblate, external_field2)

    np.testing.assert_allclose(np.abs(be1), np.abs(be2))
    np.testing.assert_allclose(np.abs(bn1), np.abs(bn2))
    np.testing.assert_allclose(np.abs(bu1), np.abs(bu2))


def test_zero_susceptibility():
    """
    Test for the case of 0 susceptibility == inducing field.
    """
    susceptibility = 0
    ellipsoid = Ellipsoid(2, 2, 1, susceptibility=susceptibility)
    coordinates = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5
    )
    external_field = hm.magnetic_angles_to_vec(55_000, 0.0, 90.0)

    be, bn, bu = ellipsoid_magnetic(coordinates, ellipsoid, external_field)

    np.testing.assert_allclose(be[0], 0)
    np.testing.assert_allclose(bn[0], 0)
    np.testing.assert_allclose(bu[0], 0)


def test_zero_field():
    """
    Test that zero field produces zero anomalies.
    """
    susceptibility = 0.01
    ellipsoid = Ellipsoid(2, 1, 1, susceptibility=susceptibility)
    coordinates = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5
    )

    external_field = (0, 0, 0)
    be, bn, bu = ellipsoid_magnetic(coordinates, ellipsoid, external_field)

    np.testing.assert_allclose(be[0], 0)
    np.testing.assert_allclose(bn[0], 0)
    np.testing.assert_allclose(bu[0], 0)


def test_mag_ext_int_boundary():
    """
    Check the boundary between internal and external field calculations is
    consistent.
    """

    a, c = 50, 60
    external_field = (55_000, 0.0, 90.0)
    susceptibility = 0.01

    ellipsoid = Ellipsoid(a, a, c, susceptibility=susceptibility)

    e = np.array([49.99, 50.00])
    n = np.array([0.0, 0.0])
    u = np.array([0.0, 0.0])
    coordinates = (e, n, u)

    be, _, _ = ellipsoid_magnetic(coordinates, ellipsoid, external_field)

    np.testing.assert_allclose(be[0], be[1], rtol=1e-7)


def test_mag_flipped_ellipsoid():
    """
    Check that rotating the ellipsoid in various ways maintains expected
    results.

    """
    a, b, c = (4, 3, 2)
    external_field = (10_000, 0, 0)
    susceptibility = 0.01

    triaxial_example = Ellipsoid(
        a, b, c, yaw=0, pitch=0, roll=0, center=(0, 0, 0), susceptibility=susceptibility
    )
    triaxial_example2 = Ellipsoid(
        a,
        b,
        c,
        yaw=180,
        pitch=180,
        roll=180,
        center=(0, 0, 0),
        susceptibility=susceptibility,
    )

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    x, y, z = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    # ignore internal field as this won't be 'flipped' in the same natr
    internal_mask = ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1
    coordinates = tuple(c[internal_mask] for c in (x, y, z))

    be1, bn1, bu1 = ellipsoid_magnetic(coordinates, triaxial_example, external_field)
    be2, bn2, bu2 = ellipsoid_magnetic(coordinates, triaxial_example2, external_field)

    np.testing.assert_allclose(np.abs(be1), np.abs(be2))
    np.testing.assert_allclose(np.abs(bn1), np.abs(bn2))
    np.testing.assert_allclose(np.abs(bu1), np.abs(bu2))


def test_euler_rotation_symmetry_mag():
    """
    Check thoroughly that euler rotations (e.g. 180 or 360 rotations) produce
    the expected result.
    """

    a, b, c = 5, 4, 3
    external_field = (55_000, 0.0, 90.0)
    susceptibility = 0.01
    coordinates = x, y, z = vd.grid_coordinates(
        region=(-5, 5, -5, 5), spacing=1.0, extra_coords=5
    )
    internal_mask = ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1
    coordinates = tuple(c[internal_mask] for c in (x, y, z))

    def check_rotation_equivalence(base_ellipsoid, rotated_ellipsoids):
        base_be, base_bn, base_bu = ellipsoid_magnetic(
            coordinates, base_ellipsoid, external_field
        )
        for rotated in rotated_ellipsoids:
            be, bn, bu = ellipsoid_magnetic(coordinates, rotated, external_field)
            np.testing.assert_allclose(np.abs(be), np.abs(base_be), rtol=1e-4)
            np.testing.assert_allclose(np.abs(bn), np.abs(base_bn), rtol=1e-4)
            np.testing.assert_allclose(np.abs(bu), np.abs(base_bu), rtol=1e-4)

    # triaxial cases
    base_tri = Ellipsoid(a, b, c, susceptibility=susceptibility)
    tri_rotated = [
        Ellipsoid(a, b, c, yaw=360, susceptibility=susceptibility),
        Ellipsoid(a, b, c, pitch=180, susceptibility=susceptibility),
        Ellipsoid(a, b, c, pitch=360, roll=360, susceptibility=susceptibility),
    ]
    check_rotation_equivalence(base_tri, tri_rotated)

    # prolate cases
    base_pro = Ellipsoid(a, b, b, susceptibility=susceptibility)
    pro_rotated = [
        Ellipsoid(a, b, b, yaw=360, susceptibility=susceptibility),
        Ellipsoid(a, b, b, pitch=180, susceptibility=susceptibility),
    ]
    check_rotation_equivalence(base_pro, pro_rotated)

    # oblate cases
    base_obl = Ellipsoid(a, a, c, susceptibility=susceptibility)
    obl_rotated = [
        Ellipsoid(a, a, c, yaw=360, susceptibility=susceptibility),
        Ellipsoid(a, a, c, pitch=180, susceptibility=susceptibility),
    ]
    check_rotation_equivalence(base_obl, obl_rotated)


class TestDemagnetizationEffects:
    """
    Test the ``get_magnetisation`` function.
    """

    @pytest.fixture(params=("oblate", "prolate", "triaxial"))
    def ellipsoid_semiaxes(self, request):
        ellipsoid_type = request.param
        match ellipsoid_type:
            case "oblate":
                a = b = 60.0
                c = 50.0
            case "prolate":
                a = 60.0
                b = c = 50.0
            case "triaxial":
                a, b, c = 70.0, 60.0, 50.0
            case _:
                raise ValueError()
        return a, b, c

    def test_demagnetization(self, ellipsoid_semiaxes):
        """
        Test demagnetization effects in ``get_magnetization``.

        The magnetization accounting with demagnetization should have a smaller
        magnitude than the magnetization without considering it.
        """
        h0_field = np.array([55_000.0, 10_000.0, -2_000.0])

        a, b, c = ellipsoid_semiaxes
        susceptibility = 0.5
        susceptibility_tensor = susceptibility * np.identity(3)

        # Compute magnetization considering demagnetization effect
        rem_mag = np.array([0, 0, 0])
        magnetization = get_magnetisation(
            a, b, c, susceptibility_tensor, h0_field, rem_mag
        )

        # Compute magnetization without considering demagnetization effect
        magnetization_no_demag = susceptibility * h0_field

        # Check that the former is smaller than the latter
        assert (magnetization**2).sum() < (magnetization_no_demag**2).sum()


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        (60.0, 60.0, 50.0),  # oblate
        (60.0, 50.0, 50.0),  # prolate
        (70.0, 60.0, 50.0),  # triaxial
        (70.0, 70.0, 70.0),  # sphere
    ],
)
class TestInternalDemagnetizationTensor:
    """
    Test properties of the demagnetization tensor on internal points.
    """

    def test_internal_demagnetization_components(self, a, b, c):
        r"""
        Test if demagnetization tensors inside the ellipsoids have all positive values.

        This guarantees that the code implements the appropriate sign convention for the
        demagnetization tensor :math:`\mathbf{N}`, defined as:

        .. math::

            \mathbf{H}(\mathbf{r}) = \mathbf{H}_0 - \mathbf{N}(\mathbf{r}) \mathbf{M}
        """
        demag_tensor = get_demagnetization_tensor_internal(a, b, c)

        # Check that the tensor is diagonal
        assert demag_tensor.ndim == 2
        diagonal = np.diagonal(demag_tensor)
        np.testing.assert_array_equal(demag_tensor, np.diag(diagonal))

        # Check that all diagonal elements are positive
        assert (diagonal > 0).all()

    def test_internal_depol_equals_1(self, a, b, c):
        """Test that the internal demagnetization tensor components sum equals 1"""
        demag_tensor = get_demagnetization_tensor_internal(a, b, c)

        # Check that the tensor is diagonal
        assert demag_tensor.ndim == 2
        diagonal = np.diagonal(demag_tensor)
        np.testing.assert_array_equal(demag_tensor, np.diag(diagonal))

        # Check that the diagonal components sum 1
        np.testing.assert_allclose(np.sum(diagonal), 1.0)


class TestMagnetizationVersusSphere:
    """
    Test if ellipsoid's magnetization approximates the one of the sphere.
    """

    # Ratio between ellipsoid's semiaxes, defined as ratio = | a - b | /  max(a, b).
    # Make it small enough so ellipsoids approximate a sphere, but not too small that
    # might trigger the fixes for numerical uncertainties.
    ratio = 1e-4
    assert ratio > SEMIAXES_RTOL

    @pytest.fixture
    def radius(self):
        """
        Sphere radius.
        """
        return 50.0

    @pytest.fixture(params=("oblate", "prolate", "triaxial"))
    def ellipsoid_semiaxes(self, radius, request):
        """
        Ellipsoid's semiaxes that approximate a sphere.
        """
        ellipsoid_type = request.param
        match ellipsoid_type:
            case "oblate":
                a = b = radius
                c = (1 - self.ratio) * b
            case "prolate":
                a = radius
                b = c = (1 - self.ratio) * a
            case "triaxial":
                a = radius
                b = (1 - self.ratio) * a
                c = (1 - 2 * self.ratio) * a
            case _:
                raise ValueError()
        return a, b, c

    def test_magnetization_vs_sphere(self, ellipsoid_semiaxes):
        """
        Test if ellipsoid's magnetization approximates the one of the sphere.
        """
        # Define moderately high susceptibility to account for demagnetization effects
        susceptibility = 0.5

        # Define arbitrary external field
        intensity, inclination, declination = 55_321, 70.2, -12.3
        b0_field = np.array(
            hm.magnetic_angles_to_vec(intensity, inclination, declination)
        )
        h0_field = b0_field / mu_0 * 1e-9  # convert to T

        # Compute magnetizations
        a, b, c = ellipsoid_semiaxes
        k_matrix = susceptibility * np.identity(3)
        rem_mag = np.array([0.0, 0.0, 0.0])
        magnetization_ellipsoid = get_magnetisation(
            a, b, c, k_matrix, h0_field, rem_mag
        )
        magnetization_sphere = get_magnetisation(a, a, a, k_matrix, h0_field, rem_mag)

        # Compare magnetization of the sphere vs magnetization of the ellipsoid
        rtol = 1e-4
        np.testing.assert_allclose(
            magnetization_ellipsoid, magnetization_sphere, rtol=rtol
        )


class TestMagneticFieldVersusSphere:
    """
    Test if magnetic field of ellipsoid approximates the one of the sphere.
    """

    # Sphere radius and susceptibility.
    radius = 50.0
    susceptibility = 0.5

    # Ratio between ellipsoid's semiaxes, defined as ratio = | a - b | /  max(a, b).
    # Make it small enough so ellipsoids approximate a sphere, but not too small that
    # might trigger the fixes for numerical uncertainties.
    ratio = 1e-4
    assert ratio > SEMIAXES_RTOL

    # Define external field
    external_field = (55_123.0, 32.0, -28.9)

    @pytest.fixture(params=[0.0, 100.0], ids=["height=0", "height=100"])
    def coordinates(self, request):
        """Sample coordinates of observation points."""
        region = (-200, 200, -200, 200)
        shape = (151, 151)
        height = request.param
        coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)
        return coordinates

    def get_ellipsoid(self, ellipsoid_type: str):
        """
        Return ellipsoid that approximates a sphere.

        Parameters
        ----------
        ellipsoid_type : {"oblate", "prolate", "triaxial"}
            Type of ellipsoid.

        Returns
        -------
        Ellipsoid
        """
        match ellipsoid_type:
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
        ellipsoid = Ellipsoid(a, b, c, susceptibility=self.susceptibility)
        return ellipsoid

    @pytest.mark.parametrize("ellipsoid_type", ["oblate", "prolate", "triaxial"])
    def test_magnetic_field_vs_sphere(self, coordinates, ellipsoid_type):
        """
        Test magnetic field of ellipsoids against the one for a sphere.
        """
        sphere = Ellipsoid(
            self.radius, self.radius, self.radius, susceptibility=self.susceptibility
        )
        b_sphere = ellipsoid_magnetic(coordinates, sphere, self.external_field)

        ellipsoid = self.get_ellipsoid(ellipsoid_type)
        b_ellipsoid = ellipsoid_magnetic(coordinates, ellipsoid, self.external_field)

        rtol = 5e-4
        for bi_sphere, bi_ellipsoid in zip(b_sphere, b_ellipsoid, strict=True):
            maxabs = vd.maxabs(bi_sphere, bi_ellipsoid)
            atol = maxabs * 5e-4
            np.testing.assert_allclose(bi_sphere, bi_ellipsoid, atol=atol, rtol=rtol)


class TestMagneticFieldVersusDipole:
    """
    Test if magnetic field of ellipsoid approximates the one of a dipole.
    """

    # Sphere radius, center, and susceptibility.
    radius = 50.0
    center = (0, 0, 0)
    susceptibility = 0.001  # use small sus to reduce demag effects

    # Ratio between ellipsoid's semiaxes, defined as ratio = | a - b | /  max(a, b).
    # Make it small enough so ellipsoids approximate a sphere, but not too small that
    # might trigger the fixes for numerical uncertainties.
    ratio = 1e-4
    assert ratio > SEMIAXES_RTOL

    # Define external field
    external_field = (55_123.0, 32.0, -28.9)

    @pytest.fixture
    def coordinates(self):
        """Sample coordinates of observation points."""
        easting = np.hstack(
            (np.linspace(-200, -self.radius, 21), np.linspace(self.radius, 200, 21))
        )
        coordinates = np.meshgrid(easting, easting, easting)
        return coordinates

    def get_ellipsoid(self, ellipsoid_type: str):
        """
        Return ellipsoid that approximates a sphere.

        Parameters
        ----------
        ellipsoid_type : {"oblate", "prolate", "triaxial"}
            Type of ellipsoid.

        Returns
        -------
        Ellipsoid
        """
        match ellipsoid_type:
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
            case "sphere":
                a = b = c = self.radius
            case _:
                raise ValueError()
        ellipsoid = Ellipsoid(a, b, c, susceptibility=self.susceptibility)
        return ellipsoid

    def get_dipole_moment(self, ellipsoid):
        """
        Convert the magnetization of an ellipsoid to the dipole magnetic moment.

        Assume the ellipsoid is close enough to a sphere for the conversion to be
        valid. Don't consider demagnetization effects.
        """
        b0_field = hm.magnetic_angles_to_vec(*self.external_field)
        h0_field = np.array(b0_field) * 1e-9 / mu_0  # convert to SI units
        return 4 / 3 * np.pi * ellipsoid.a**3 * ellipsoid.susceptibility * h0_field

    @pytest.mark.parametrize(
        "ellipsoid_type", ["oblate", "prolate", "triaxial", "sphere"]
    )
    def test_magnetic_field_vs_dipole(self, coordinates, ellipsoid_type):
        """
        Test magnetic field of ellipsoids against the one of a dipole.
        """
        # Forward model the magnetic field of the ellipsoid
        ellipsoid = self.get_ellipsoid(ellipsoid_type)
        b_ellipsoid = ellipsoid_magnetic(coordinates, ellipsoid, self.external_field)

        # Forward model the magnetic field of the dipole
        dipole_moment = self.get_dipole_moment(ellipsoid)
        b_dipole = hm.dipole_magnetic(
            coordinates, ellipsoid.center, dipole_moment, field="b"
        )

        rtol = 5e-4
        for bi_dipole, bi_ellipsoid in zip(b_dipole, b_ellipsoid, strict=True):
            maxabs = vd.maxabs(bi_dipole, bi_ellipsoid)
            atol = maxabs * 5e-4
            np.testing.assert_allclose(bi_dipole, bi_ellipsoid, rtol=rtol, atol=atol)


class TestSymmetryOnRotations:
    """
    Test symmetries in the magnetic field after rotations are applied.
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
        match ellipsoid_type:
            case "oblate":
                a = b = semimajor
                c = semiminor
            case "prolate":
                a = semimajor
                b = c = semiminor
            case "triaxial":
                a, b, c = semimajor, semimiddle, semiminor
            case _:
                raise ValueError()

        ellipsoid = Ellipsoid(a, b, c, yaw=yaw, pitch=pitch, roll=roll, center=center)
        return ellipsoid

    @pytest.mark.parametrize("magnetization_type", ["induced", "remanent", "both"])
    def test_symmetry_when_flipping(self, ellipsoid, magnetization_type):
        """
        Test symmetry of magnetic field when flipping the ellipsoid.

        Rotate the ellipsoid so the geometry is preserved. The magnetic field generated
        by the ellipsoid should be the same as before the rotation.

        Since the remanent magnetization vector is defined in the global coordinate
        system, it won't rotate with the ellipsoid.
        """
        # Define observation points
        coordinates = vd.grid_coordinates(
            region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
        )

        # Define physical properties
        external_field = (55_000, -71, 15)
        magnetization = (
            (400, 21, -8) if magnetization_type in ("remanent", "both") else (0, 0, 0)
        )
        susceptibility = 0.1 if magnetization_type in ("induced", "both") else 0.0

        # Assign physical properties to the ellipsoid
        ellipsoid.susceptibility = susceptibility
        ellipsoid.remanent_mag = magnetization

        # Generate a flipped copy of the ellipsoid
        ellipsoid_flipped = self.flip_ellipsoid(copy(ellipsoid))

        # Compute magnetic fields
        b_field, b_field_flipped = tuple(
            ellipsoid_magnetic(coordinates, ell, external_field)
            for ell in (ellipsoid, ellipsoid_flipped)
        )

        # Check that the B field is the same for original and flipped ellipsoids
        for i in range(3):
            np.testing.assert_allclose(b_field[i], b_field_flipped[i])


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
                a=60,
                b=60,
                c=20,
                yaw=30.2,
                pitch=-23,
                center=(-10.0, 20.0, -10.0),
            ),
            Ellipsoid(
                a=40,
                b=15,
                c=15,
                yaw=170.2,
                pitch=71,
                center=(15.0, 0.0, -40.0),
            ),
            Ellipsoid(
                a=60,
                b=18,
                c=15,
                yaw=272.1,
                pitch=43,
                roll=98,
                center=(0.0, 20.0, -30.0),
            ),
        ]
        return ellipsoids

    @pytest.mark.parametrize("sus_type", ["isotropic", "anisotropic"])
    def test_multiple_ellipsoids_susceptibilities(
        self, coordinates, ellipsoids, sus_type
    ):
        """
        Run forward function with multiple ellipsoids (only with susceptibilities).
        """
        # Assign susceptibilities to ellipsoids
        if sus_type == "isotropic":
            susceptibilities = [0.1, 0.01, 0.05]
        else:
            sus_tensor = np.random.default_rng(seed=42).uniform(size=(3, 3))
            susceptibilities = [0.1, 0.01, sus_tensor]
        for ellipsoid, susceptibility in zip(ellipsoids, susceptibilities, strict=True):
            ellipsoid.susceptibility = susceptibility

        # Define external field
        external_field = (55_000, -15, 65)

        # Compute magnetic field
        bx, by, bz = ellipsoid_magnetic(
            coordinates,
            ellipsoids,
            external_field,
        )

        # Compute expected arrays
        bx_expected, by_expected, bz_expected = tuple(
            np.zeros_like(coordinates[0]) for _ in range(3)
        )
        for ellipsoid in ellipsoids:
            bx_i, by_i, bz_i = ellipsoid_magnetic(
                coordinates, ellipsoid, external_field
            )
            bx_expected += bx_i
            by_expected += by_i
            bz_expected += bz_i

        # Check if fields are the same
        np.testing.assert_allclose(bx, bx_expected)
        np.testing.assert_allclose(by, by_expected)
        np.testing.assert_allclose(bz, bz_expected)

    def test_multiple_ellipsoids_remanence(self, coordinates, ellipsoids):
        """
        Run forward function with multiple ellipsoids with remanence.
        """
        # Assign remanent magnetizations to ellipsoids
        remanent_mags = [
            [1.0, 2.0, 3.0],
            [5.0, -1.0, -3.0],
            [10.0, 3.0, -5.0],
        ]
        for ellipsoid, remanent_mag in zip(ellipsoids, remanent_mags, strict=True):
            ellipsoid.remanent_mag = remanent_mag

        # Compute magnetic field
        external_field = (55_000, -15, 65)
        bx, by, bz = ellipsoid_magnetic(
            coordinates,
            ellipsoids,
            external_field,
        )

        # Compute expected arrays
        bx_expected, by_expected, bz_expected = tuple(
            np.zeros_like(coordinates[0]) for _ in range(3)
        )
        for ellipsoid in ellipsoids:
            bx_i, by_i, bz_i = ellipsoid_magnetic(
                coordinates,
                ellipsoid,
                external_field,
            )
            bx_expected += bx_i
            by_expected += by_i
            bz_expected += bz_i

        # Check if fields are the same
        np.testing.assert_allclose(bx, bx_expected)
        np.testing.assert_allclose(by, by_expected)
        np.testing.assert_allclose(bz, bz_expected)


class TestNoMagnetic:
    """Test warning when ellipsoid has no susceptibility nor remanent magnetization."""

    def test_warning(self):
        """
        Test warning about ellipsoid with no susceptibility nor remanence being skipped.
        """
        coordinates = (0.0, 0.0, 0.0)
        a, b, c = 50.0, 35.0, 25.0
        ellipsoid = Ellipsoid(a, b, c)
        external_field = (55_000.0, 13, 71)

        msg = re.escape(
            f"Ellipsoid {ellipsoid} doesn't have a susceptibility nor a "
            "remanent_mag value. It will be skipped."
        )
        with pytest.warns(NoPhysicalPropertyWarning, match=msg):
            bx, by, bz = ellipsoid_magnetic(coordinates, ellipsoid, external_field)

        # Check the magnetic field components are zero
        for b_component in (bx, by, bz):
            assert b_component == 0.0


class TestTriaxialOnLimits:
    """
    Test a triaxial ellipsoid vs oblate and prolate ellipsoids.

    Test the magnetic fields of a triaxial ellipsoid that approximates an oblate and
    a prolate one against their magnetic fields.
    """

    semimajor = 50.0
    external_field = (55_000, 71, 18)
    atol_ratio = 1e-6
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
        susceptibility = 0.2
        triaxial = Ellipsoid(a, b, c, susceptibility=susceptibility)
        prolate = Ellipsoid(a, b, b, susceptibility=susceptibility)

        b_triaxial, b_prolate = tuple(
            ellipsoid_magnetic(coordinates, ell, external_field=self.external_field)
            for ell in (triaxial, prolate)
        )

        for bi_triaxial, bi_prolate in zip(b_triaxial, b_prolate, strict=True):
            atol = self.atol_ratio * vd.maxabs(bi_prolate)
            np.testing.assert_allclose(
                bi_triaxial, bi_prolate, atol=atol, rtol=self.rtol
            )

    def test_triaxial_vs_oblate(self):
        """Compare triaxial with oblate ellipsoid."""
        coordinates = self.get_coordinates()
        a = self.semimajor
        b = a - 1e-4
        c = 20.0
        susceptibility = 0.2
        triaxial = Ellipsoid(a, b, c, susceptibility=susceptibility)
        oblate = Ellipsoid(a, a, c, susceptibility=susceptibility)

        b_triaxial, b_oblate = tuple(
            ellipsoid_magnetic(coordinates, ell, external_field=self.external_field)
            for ell in (triaxial, oblate)
        )

        for bi_triaxial, bi_oblate in zip(b_triaxial, b_oblate, strict=True):
            atol = self.atol_ratio * vd.maxabs(bi_oblate)
            np.testing.assert_allclose(
                bi_triaxial, bi_oblate, atol=atol, rtol=self.rtol
            )


class TestSemiaxesArbitraryOrder:
    """
    Test magnetic fields when defining the same ellipsoids with different orders of the
    semiaxes plus needed rotation angles.
    """

    atol_ratio = 0.0
    rtol = 1e-7
    external_field = (55_000, 17, -21)

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
        susceptibility = 0.2
        rem_mag = np.array([1.0, 2.0, -3.0])
        ellipsoid = Ellipsoid(
            a, b, c, susceptibility=susceptibility, remanent_mag=rem_mag
        )
        ellipsoid_rotated = Ellipsoid(
            *semiaxes_sorted,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            susceptibility=susceptibility,
            remanent_mag=rem_mag,
        )

        coordinates = self.get_coordinates(semiaxes_sorted[0])
        g_field = ellipsoid_magnetic(
            coordinates, ellipsoid, external_field=self.external_field
        )
        g_rotated = ellipsoid_magnetic(
            coordinates, ellipsoid_rotated, external_field=self.external_field
        )

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
        susceptibility = 0.2
        rem_mag = np.array([1.0, 2.0, -3.0])
        ellipsoid = Ellipsoid(
            a, b, c, susceptibility=susceptibility, remanent_mag=rem_mag
        )
        ellipsoid_rotated = Ellipsoid(
            *semiaxes_sorted,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            susceptibility=susceptibility,
            remanent_mag=rem_mag,
        )

        coordinates = self.get_coordinates(semiaxes_sorted[0])
        g_field = ellipsoid_magnetic(
            coordinates, ellipsoid, external_field=self.external_field
        )
        g_rotated = ellipsoid_magnetic(
            coordinates, ellipsoid_rotated, external_field=self.external_field
        )

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
        susceptibility = 0.2
        rem_mag = np.array([1.0, 2.0, -3.0])
        ellipsoid = Ellipsoid(
            a, b, c, susceptibility=susceptibility, remanent_mag=rem_mag
        )
        ellipsoid_rotated = Ellipsoid(
            *semiaxes_sorted,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            susceptibility=susceptibility,
            remanent_mag=rem_mag,
        )

        coordinates = self.get_coordinates(semiaxes_sorted[0])
        g_field = ellipsoid_magnetic(
            coordinates, ellipsoid, external_field=self.external_field
        )
        g_rotated = ellipsoid_magnetic(
            coordinates, ellipsoid_rotated, external_field=self.external_field
        )

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
    susceptibility = 0.1
    external_field = (55_000, 17, 21)

    def get_coordinates(self, azimuth=45, polar=45):
        """
        Generate coordinates of observation points along a certain direction.
        """
        r = np.linspace(1e-3 * self.semimajor, 5.5 * self.semimajor, 501)
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
            susceptibility=self.susceptibility,
        )

    @pytest.fixture
    def oblate(self):
        return Ellipsoid(
            self.semimajor,
            self.semimajor,
            self.semiminor,
            center=self.center,
            susceptibility=self.susceptibility,
        )

    def test_triaxial_vs_prolate(self, prolate):
        """
        Test magnetic field of triaxial ellipsoid that is almost a prolate.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-14

        a = self.semimajor
        b = self.semiminor
        c = (1 - ratio) * b
        ellipsoid = Ellipsoid(
            a, b, c, center=self.center, susceptibility=self.susceptibility
        )
        be_prolate, bn_prolate, bu_prolate = ellipsoid_magnetic(
            coordinates, prolate, self.external_field
        )
        be_ell, bn_ell, bu_ell = ellipsoid_magnetic(
            coordinates, ellipsoid, self.external_field
        )

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(be_ell, be_prolate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(bn_ell, bn_prolate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(bu_ell, bu_prolate, rtol=rtol, atol=atol)

    def test_triaxial_vs_oblate(self, oblate):
        """
        Test magnetic field of triaxial ellipsoid that is almost a oblate.
        """
        coordinates = self.get_coordinates()

        # Semiaxes ratio. Sufficiently small to trigger the numerical instabilities
        ratio = 1e-14

        a = self.semimajor
        b = (1 - ratio) * a
        c = self.semiminor
        ellipsoid = Ellipsoid(
            a, b, c, center=self.center, susceptibility=self.susceptibility
        )
        be_oblate, bn_oblate, bu_oblate = ellipsoid_magnetic(
            coordinates, oblate, self.external_field
        )
        be_ell, bn_ell, bu_ell = ellipsoid_magnetic(
            coordinates, ellipsoid, self.external_field
        )

        rtol, atol = 1e-7, 1e-8
        np.testing.assert_allclose(be_ell, be_oblate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(bn_ell, bn_oblate, rtol=rtol, atol=atol)
        np.testing.assert_allclose(bu_ell, bu_oblate, rtol=rtol, atol=atol)


class TestAnisotropy:
    """
    Test behaviour of susceptibility as a tensor.
    """

    def test_invariance_semiaxes_order(self):
        """
        Test if fields are invariant under change of semiaxes order.

        Check if the susceptibility tensor is correctly rotated using the semiaxes
        rotation matrix.
        """
        region = (-50, 50, -50, 50)
        coordinates = vd.grid_coordinates(region, shape=(151, 151), extra_coords=0)
        inducing_field = np.array([0, 0, -20_000])  # pointing on z
        # Define susceptibility as a tensor with only component on the vertical axes
        # of the local coordinate system.
        susceptibility = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        )
        semiaxes = 5.002, 5.001, 5.0
        ellipsoids = [
            hm.Ellipsoid(a, b, c, center=(0, 0, -20), susceptibility=susceptibility)
            for a, b, c in itertools.permutations(semiaxes)
        ]

        b_fields = np.array(
            [
                hm.ellipsoid_magnetic(
                    coordinates, ellipsoid, inducing_field=inducing_field
                )
                for ellipsoid in ellipsoids
            ]
        )

        atol_ratio = 1e-5
        for b_field in b_fields[1:]:
            for i in range(3):
                maxabs = vd.maxabs(b_field[i])
                np.testing.assert_allclose(
                    b_fields[0, i], b_field[i], rtol=5e-4, atol=atol_ratio * maxabs
                )
