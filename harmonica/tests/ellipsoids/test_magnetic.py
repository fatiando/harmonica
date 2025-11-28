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

    # Difference between ellipsoid's semiaxes.
    # It should be small compared to the sphere radius, so the ellipsoid approximates
    # a sphere.
    delta = 1e-2

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
        a = radius
        ellipsoid_type = request.param
        match ellipsoid_type:
            case "oblate":
                b = a
                c = a - self.delta
            case "prolate":
                b = c = a - self.delta
            case "triaxial":
                b = a - self.delta
                c = a - 2 * self.delta
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

    # Sphere radius, center, and susceptibility.
    radius = 50.0
    center = (0, 0, 0)
    susceptibility = 0.5

    # Difference between ellipsoid's semiaxes.
    # It should be small compared to the sphere radius, so the ellipsoid approximates
    # a sphere.
    delta = 0.001

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
                c = a - self.delta
            case "prolate":
                a = self.radius
                b = c = a - self.delta
            case "triaxial":
                a = self.radius
                b = a - self.delta
                c = a - 2 * self.delta
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
            atol = maxabs * 1e-4
            np.testing.assert_allclose(bi_sphere, bi_ellipsoid, atol=atol, rtol=rtol)


class TestMagneticFieldVersusDipole:
    """
    Test if magnetic field of ellipsoid approximates the one of a dipole.
    """

    # Sphere radius, center, and susceptibility.
    radius = 50.0
    center = (0, 0, 0)
    susceptibility = 0.001  # use small sus to reduce demag effects

    # Difference between ellipsoid's semiaxes.
    # It should be small compared to the sphere radius, so the ellipsoid approximates
    # a sphere.
    delta = 0.001

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
                c = a - self.delta
            case "prolate":
                a = self.radius
                b = c = a - self.delta
            case "triaxial":
                a = self.radius
                b = a - self.delta
                c = a - 2 * self.delta
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
            atol = maxabs * 1e-4
            np.testing.assert_allclose(bi_dipole, bi_ellipsoid, atol=atol, rtol=rtol)


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

        # Check the gravity acceleration components are zero
        for b_component in (bx, by, bz):
            assert b_component == 0.0
