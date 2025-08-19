import numpy as np
import verde as vd
from scipy.constants import mu_0

import harmonica as hm

from .._forward.create_ellipsoid import (
    OblateEllipsoid,
    ProlateEllipsoid,
    TriaxialEllipsoid,
)
from .._forward.ellipsoid_magnetics import (
    _depol_oblate_int,
    _depol_prolate_int,
    _depol_triaxial_int,
    ellipsoid_magnetics,
)
from .._forward.utils_ellipsoids import _get_v_as_euler, _sphere_magnetic


def test_likeness_to_sphere():
    """Using a, b, c as almost equal, compare how the close the ellipsoids
    match the dipole-sphere magnetic approximation for low susceptabilities.
    At higher susceptibilities, the self-demag will make the ellipsoid
    deviate."""

    # create field
    k = [0.01, 0.001, 0.0001]
    b0 = np.array(hm.magnetic_angles_to_vec(55_000, 0.0, 90.0))
    h0_am = np.array(b0 * 1e-9 / mu_0)
    m = [k * h0_am for k in k]

    # create coords
    easting = np.linspace(0, 2 * 60, 50)
    northing, upward = np.zeros_like(easting), np.zeros_like(easting)
    coordinates = tuple(np.atleast_2d(c) for c in (easting, northing, upward))

    # create ellipsoids
    pro_ellipsoid = ProlateEllipsoid(
        a=60, b=59.99, yaw=0, pitch=0, centre=(0, 0, 0)
    )
    tri_ellipsoid = TriaxialEllipsoid(
        a=60, b=59.999, c=59.998, yaw=0, pitch=0, roll=0, centre=(0, 0, 0)
    )
    obl_ellipsoid = OblateEllipsoid(
        a=59.99, b=60, yaw=0, pitch=0, centre=(0, 0, 0)
    )

    for indx, k in enumerate(k):

        # ellipsoids
        be_pro, _, _ = ellipsoid_magnetics(
            coordinates, pro_ellipsoid, k, (55_000, 0.0, 90.0), field="b"
        )
        be_pro = be_pro.ravel()
        be_tri, _, _ = ellipsoid_magnetics(
            coordinates, tri_ellipsoid, k, (55_000, 0.0, 90.0), field="b"
        )
        be_tri = be_tri.ravel()
        be_obl, _, _ = ellipsoid_magnetics(
            coordinates, obl_ellipsoid, k, (55_000, 0.0, 90.0), field="b"
        )
        be_obl = be_obl.ravel()

        # sphere
        b_e, b_n, b_u = _sphere_magnetic(
            coordinates, radius=60, center=(0, 0, 0), magnetization=m[indx]
        )
        b_e = b_e.ravel()

        # test similarity
        np.testing.assert_allclose(be_pro, b_e, rtol=1e-2)

        np.testing.assert_allclose(be_tri, b_e, rtol=1e-2)

        np.testing.assert_allclose(be_obl, b_e, rtol=1e-2)


def test_euler_returns():
    """Check the euler returns are exact"""
    r0 = _get_v_as_euler(0, 0, 0)
    r360 = _get_v_as_euler(360, 0, 0)
    assert np.allclose(r0, r360)


def test_magnetic_symmetry():
    """
    Check the symmetry of magentic calculations at surfaces above and below
    the body.
    """
    a, b, c = (4, 3, 2)  # triaxial ellipsoid
    yaw, pitch, roll = (0, 0, 0)
    external_field = (10_000, 0, 0)
    susceptabililty = 0.1
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
        coordinates,
        triaxial_example,
        susceptabililty,
        external_field,
        field="b",
    )
    be2, bn2, bu2 = ellipsoid_magnetics(
        coordinates2,
        triaxial_example2,
        susceptabililty,
        external_field,
        field="b",
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
    yaw, pitch = 0, 0

    external_field1 = np.array((55_000, 0.0, 90.0))
    external_field2 = -external_field1
    susceptabililty = 0.1
    oblate_example = OblateEllipsoid(a, b, yaw, pitch, (0, 0, 0))

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    be1, bn1, bu1 = ellipsoid_magnetics(
        coordinates,
        oblate_example,
        susceptabililty,
        external_field1,
        field="b",
    )
    be2, bn2, bu2 = ellipsoid_magnetics(
        coordinates,
        oblate_example,
        susceptabililty,
        external_field2,
        field="b",
    )

    np.testing.assert_allclose(np.abs(be1), np.abs(be2))
    np.testing.assert_allclose(np.abs(bn1), np.abs(bn2))
    np.testing.assert_allclose(np.abs(bu1), np.abs(bu2))


def test_zero_susceptability():
    """
    Test for the case of 0 susceptabililty == inducing field.
    """

    a, b = 1, 2
    susceptabililty = 0
    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    coordinates = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5
    )
    h0 = hm.magnetic_angles_to_vec(55_000, 0.0, 90.0)

    be, bn, bu = ellipsoid_magnetics(
        coordinates, ellipsoid, susceptabililty, h0, field="b"
    )

    np.testing.assert_allclose(be[0], 0)
    np.testing.assert_allclose(bn[0], 0)
    np.testing.assert_allclose(bu[0], 0)


def test_zero_field():
    """
    Test that zero field produces zero anomalies.
    """

    a, b = 1, 2
    external_field = np.array([0, 0, 0])
    susceptabililty = 0.01

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    coordinates = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5
    )

    be, bn, bu = ellipsoid_magnetics(
        coordinates, ellipsoid, susceptabililty, external_field, field="b"
    )

    np.testing.assert_allclose(be[0], 0)
    np.testing.assert_allclose(bn[0], 0)
    np.testing.assert_allclose(bu[0], 0)


def test_mag_ext_int_boundary():
    """
    Check the boundary between internal and external field calculations is
    consistent.
    """

    a, b = 50, 60
    external_field = (55_000, 0.0, 90.0)
    susceptabililty = 0.01

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))

    e = np.array([49.99, 50.00])
    n = np.array([0.0, 0.0])
    u = np.array([0.0, 0.0])
    coordinates = (e, n, u)

    be, bn, bu = ellipsoid_magnetics(
        coordinates, ellipsoid, susceptabililty, external_field, field="b"
    )

    # ideally the tolerances are lower for these - issue created
    np.testing.assert_allclose(be[0], be[1], rtol=1e-4)


def test_mag_flipped_ellipsoid():
    """
    Check that rotating the ellipsoid in various ways maintains expected
    results.

    """
    a, b, c = (4, 3, 2)
    external_field = (10_000, 0, 0)
    susceptabililty = 0.01

    triaxial_example = TriaxialEllipsoid(
        a, b, c, yaw=0, pitch=0, roll=0, centre=(0, 0, 0)
    )
    triaxial_example2 = TriaxialEllipsoid(
        a, b, c, yaw=180, pitch=180, roll=180, centre=(0, 0, 0)
    )

    # define observation points (2D grid) at surface height (z axis,
    # 'Upward') = 5
    x, y, z = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    # ignore internal field as this won't be 'flipped' in the same natr
    internal_mask = ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1
    coordinates = tuple(c[internal_mask] for c in (x, y, z))

    be1, bn1, bu1 = ellipsoid_magnetics(
        coordinates,
        triaxial_example,
        susceptabililty,
        external_field,
        field="b",
    )
    be2, bn2, bu2 = ellipsoid_magnetics(
        coordinates,
        triaxial_example2,
        susceptabililty,
        external_field,
        field="b",
    )

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
    susceptabililty = 0.01
    coordinates = x, y, z = vd.grid_coordinates(
        region=(-5, 5, -5, 5), spacing=1.0, extra_coords=5
    )
    internal_mask = ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1
    coordinates = tuple(c[internal_mask] for c in (x, y, z))

    def check_rotation_equivalence(base_ellipsoid, rotated_ellipsoids):
        base_be, base_bn, base_bu = ellipsoid_magnetics(
            coordinates, base_ellipsoid, susceptabililty, external_field
        )
        for rotated in rotated_ellipsoids:
            be, bn, bu = ellipsoid_magnetics(
                coordinates, rotated, susceptabililty, external_field
            )
            np.testing.assert_allclose(np.abs(be), np.abs(base_be), rtol=1e-4)
            np.testing.assert_allclose(np.abs(bn), np.abs(base_bn), rtol=1e-4)
            np.testing.assert_allclose(np.abs(bu), np.abs(base_bu), rtol=1e-4)

    # triaxial cases
    base_tri = TriaxialEllipsoid(
        a, b, c, yaw=0, pitch=0, roll=0, centre=(0, 0, 0)
    )
    tri_rotated = [
        TriaxialEllipsoid(a, b, c, yaw=360, pitch=0, roll=0, centre=(0, 0, 0)),
        TriaxialEllipsoid(a, b, c, yaw=0, pitch=180, roll=0, centre=(0, 0, 0)),
        TriaxialEllipsoid(
            a, b, c, yaw=0, pitch=360, roll=360, centre=(0, 0, 0)
        ),
    ]
    check_rotation_equivalence(base_tri, tri_rotated)

    # prolate cases
    base_pro = ProlateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    pro_rotated = [
        ProlateEllipsoid(a, b, yaw=360, pitch=0, centre=(0, 0, 0)),
        ProlateEllipsoid(a, b, yaw=0, pitch=180, centre=(0, 0, 0)),
    ]
    check_rotation_equivalence(base_pro, pro_rotated)

    # oblate cases
    base_obl = OblateEllipsoid(b, a, yaw=0, pitch=0, centre=(0, 0, 0))
    obl_rotated = [
        OblateEllipsoid(b, a, yaw=360, pitch=0, centre=(0, 0, 0)),
        OblateEllipsoid(b, a, yaw=0, pitch=180, centre=(0, 0, 0)),
    ]
    check_rotation_equivalence(base_obl, obl_rotated)


def test_internal_depol_equals_1():
    """Test that the internal depol tensor component sum equals 1"""

    onxx, onyy, onzz = _depol_oblate_int(3, 5, 5)
    np.testing.assert_allclose((onxx + onyy + onzz), 1)

    pnxx, pnyy, pnzz = _depol_prolate_int(5, 3, 3)
    np.testing.assert_allclose((pnxx + pnyy + pnzz), 1)

    tnxx, tnyy, tnzz = _depol_triaxial_int(5, 4, 3)
    np.testing.assert_allclose((tnxx + tnyy + tnzz), 1)
