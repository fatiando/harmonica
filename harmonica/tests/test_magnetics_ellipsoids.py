import numpy as np
import verde as vd
from scipy.constants import mu_0

from .._forward.create_ellipsoid import (
    OblateEllipsoid,
    ProlateEllipsoid,
    TriaxialEllipsoid,
)
from .._forward.ellipsoid_magnetics import ellipsoid_magnetics


def test_magnetic_symmetry():
    """
    Check the symmetry of magentic calculations at surfaces above and below
    the body.
    """
    a, b, c = (4, 3, 2)  # triaxial ellipsoid
    yaw = 0
    pitch = 0
    roll = 0
    H0 = np.array([5, 5, 5])
    triaxial_example = TriaxialEllipsoid(a, b, c, yaw, pitch, roll, (0, 0, 0))
    triaxial_example2 = TriaxialEllipsoid(a, b, c, yaw, pitch, roll, (0, 0, 0))

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )
    coordinates2 = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=-5
    )

    be1, bn1, bu1 = ellipsoid_magnetics(
        coordinates, triaxial_example, 0.1, H0, field="b"
    )
    be2, bn2, bu2 = ellipsoid_magnetics(
        coordinates2, triaxial_example2, 0.1, H0, field="b"
    )

    np.testing.assert_allclose(np.abs(be1), np.flip(np.abs(be2)))
    np.testing.assert_allclose(np.abs(bn1), np.flip(np.abs(bn2)))
    np.testing.assert_allclose(np.abs(bu1), np.flip(np.abs(bu2)))


def test_flipped_h0():
    """
    Check that reversing the magentising field produces the same (reversed)
    field.
    """

    a, b = (2, 4)  # triaxial ellipsoid
    yaw = 0
    pitch = 0
    H01 = np.array([5, 5, 5])
    H02 = np.array([-5, -5, -5])
    oblate_example = OblateEllipsoid(a, b, yaw, pitch, (0, 0, 0))

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    be1, bn1, bu1 = ellipsoid_magnetics(
        coordinates, oblate_example, 0.1, H01, field="b"
    )
    be2, bn2, bu2 = ellipsoid_magnetics(
        coordinates, oblate_example, 0.1, H02, field="b"
    )

    np.testing.assert_allclose(np.abs(be1), np.abs(be2))
    np.testing.assert_allclose(np.abs(bn1), np.abs(bn2))
    np.testing.assert_allclose(np.abs(bu1), np.abs(bu2))


def test_zero_susceptability():
    """
    Test for the case of 0 susceptabililty == inducing field.
    """

    a, b = 1, 2
    H0 = np.array([10, 0, 0])
    k = 0

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    coordinates = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5
    )

    be, bn, bu = ellipsoid_magnetics(coordinates, ellipsoid, k, H0, field="b")

    np.testing.assert_allclose(be[0], 1e9 * mu_0 * H0[0])
    np.testing.assert_allclose(bn[0], 1e9 * mu_0 * H0[1])
    np.testing.assert_allclose(bu[0], 1e9 * mu_0 * H0[2])


def test_zero_field():
    """
    Test that zero field produces zero anomalies.
    """

    a, b = 1, 2
    H0 = np.array([0, 0, 0])
    k = 0.1

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    coordinates = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5
    )

    be, bn, bu = ellipsoid_magnetics(coordinates, ellipsoid, k, H0, field="b")

    np.testing.assert_allclose(be[0], 0)
    np.testing.assert_allclose(bn[0], 0)
    np.testing.assert_allclose(bu[0], 0)


def test_mag_ext_int_boundary():
    """
    Check the boundary between internal and external field calculations is
    consistent.
    """

    # aribtrary parameters
    a, b = 1, 2
    H0 = np.array([10.0, 0.0, 0.0])
    k = 0.1
    a, b = 50, 60
    H0 = np.array([0, 0, 10])
    k = 0.1

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))

    e = np.array([[49.9999999, 50.0000001]])
    n = np.array([[0.0, 0.0]])
    u = np.array([[0.0, 0.0]])
    coordinates = (e, n, u)

    be, bn, bu = ellipsoid_magnetics(coordinates, ellipsoid, k, H0, field="b")

    # ideally the tolerances are lower for these - issue created
    np.testing.assert_allclose(be[0, 0], be[0, 1], rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(bn[0, 0], bn[0, 1], rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(bu[0, 0], bu[0, 1], rtol=5e-2, atol=5e-2)


def test_mag_flipped_ellipsoid():
    """
    Check that rotating the ellipsoid in various ways maintains expected
    results.

    """
    a, b, c = (4, 3, 2)
    H0 = np.array([5, 5, 5])
    triaxial_example = TriaxialEllipsoid(
        a, b, c, yaw=0, pitch=0, roll=0, centre=(0, 0, 0)
    )
    triaxial_example2 = TriaxialEllipsoid(
        a, b, c, yaw=180, pitch=180, roll=180, centre=(0, 0, 0)
    )

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    be1, bn1, bu1 = ellipsoid_magnetics(
        coordinates, triaxial_example, 0.1, H0, field="b"
    )
    be2, bn2, bu2 = ellipsoid_magnetics(
        coordinates, triaxial_example2, 0.1, H0, field="b"
    )

    np.testing.assert_allclose(np.abs(be1), np.abs(be2))
    np.testing.assert_allclose(np.abs(bn1), np.abs(bn2))
    np.testing.assert_allclose(np.abs(bu1), np.abs(bu2))


def test_mag_symmetry_across_N_axis():
    """
    With no rotation of the ellipsoid and an external field aligned with the
    axis, check that the symmetry of the returned magnetic field is reflected
    across the y (northing) axis.

    """

    a, b, c = (4, 3, 2)
    H0 = np.array([0, 0, 5])
    triaxial_example = TriaxialEllipsoid(
        a, b, c, yaw=0, pitch=0, roll=0, centre=(0, 0, 0)
    )

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=0
    )

    be1, bn1, bu1 = ellipsoid_magnetics(
        coordinates, triaxial_example, 0.1, H0, field="b"
    )

    # check symmetry across the easting axis (rows)
    left = be1[: be1.shape[1] // 2, :]
    right = be1[-1: -(be1.shape[1] // 2) - 1: -1, :]  # reversed columns

    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)


def test_euler_rotation_symmetry_mag():
    """
    Check thoroughly that euler rotations (e.g. 180 or 360 rotations) produce
    the expected result.
    """

    a, b, c = 5, 4, 3
    H0 = np.array([10.0, 0.0, 0.0])
    k = 0.1
    coordinates = vd.grid_coordinates(
        region=(-5, 5, -5, 5), spacing=1.0, extra_coords=5
    )

    def check_rotation_equivalence(base_ellipsoid, rotated_ellipsoids):
        base_be, base_bn, base_bu = ellipsoid_magnetics(
            coordinates, base_ellipsoid, k, H0
        )
        for rotated in rotated_ellipsoids:
            be, bn, bu = ellipsoid_magnetics(coordinates, rotated, k, H0)
            np.testing.assert_allclose(be, base_be, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(bn, base_bn, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(bu, base_bu, rtol=1e-5, atol=1e-8)

    # triaxial cases
    base_tri = TriaxialEllipsoid(a, b, c, yaw=0, pitch=0, roll=0,
                                 centre=(0, 0, 0))
    tri_rotated = [
        TriaxialEllipsoid(a, b, c, yaw=360, pitch=0, roll=0, centre=(0, 0, 0)),
        TriaxialEllipsoid(a, b, c, yaw=180, pitch=180, roll=0,
                          centre=(0, 0, 0)),
        TriaxialEllipsoid(a, b, c, yaw=0, pitch=360, roll=360,
                          centre=(0, 0, 0)),
    ]
    check_rotation_equivalence(base_tri, tri_rotated)

    # prolate cases
    base_pro = ProlateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    pro_rotated = [
        ProlateEllipsoid(a, b, yaw=360, pitch=0, centre=(0, 0, 0)),
        ProlateEllipsoid(a, b, yaw=180, pitch=180, centre=(0, 0, 0)),
    ]
    check_rotation_equivalence(base_pro, pro_rotated)

    # oblate cases
    base_obl = OblateEllipsoid(b, a, yaw=0, pitch=0, centre=(0, 0, 0))
    obl_rotated = [
        OblateEllipsoid(b, a, yaw=360, pitch=0, centre=(0, 0, 0)),
        OblateEllipsoid(b, a, yaw=180, pitch=180, centre=(0, 0, 0)),
    ]
    check_rotation_equivalence(base_obl, obl_rotated)
