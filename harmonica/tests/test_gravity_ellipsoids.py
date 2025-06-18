import numpy as np
import verde as vd
from choclo.point import gravity_u as pointgrav

from .._forward.create_ellipsoid import (
    OblateEllipsoid,
    ProlateEllipsoid,
    TriaxialEllipsoid,
)
from .._forward.ellipsoid_gravity import (
    _get_gravity_oblate,
    _get_gravity_prolate,
    _get_gravity_triaxial,
    ellipsoid_gravity,
)


def test_degenerate_ellipsoid_cases():
    """
    Test cases where the ellipsoid axes lengths are close to the boundary of
    accepted values.

    """
    tri = TriaxialEllipsoid(5, 4.99999999, 4.99999998, 0, 0, 0, (0, 0, 0))
    pro = ProlateEllipsoid(5, 4.99999999, 0, 0, (0, 0, 0))
    obl = OblateEllipsoid(4.99999999, 5, 0, 0, (0, 0, 0))

    coordinates = vd.grid_coordinates(
        region=(-20, 20, -20, 20), spacing=0.5, extra_coords=5
    )

    _, _, gu1 = ellipsoid_gravity(coordinates, tri, 2000, field="g")
    _, _, gu2 = ellipsoid_gravity(coordinates, pro, 2000, field="g")
    _, _, gu3 = ellipsoid_gravity(coordinates, obl, 2000, field="g")


def test_ellipsoid_at_distance():
    """

    To test that the triaxial ellipsoid function produces the same
    result as the scipy point mass for spherical bodies at distance.

    """

    dg1, dg2, dg3 = _get_gravity_triaxial(0, 0, 100, 3, 2, 1, density=1000)
    mass = 1000 * 4 / 3 * np.pi * 3 * 2 * 1
    point_grav = pointgrav(0, 0, 100, 0, 0, 0, mass)

    assert np.allclose(dg3, point_grav)


def test_symmetry_at_surface():
    """

    Test that the gravity anomaly produced shows symmetry across the axes.
    E.g., a surface of ellipsoid orientated to global coordinate system would
    produce an equal but opposite anaomly at surface z=5 and surface z=-5.

    """

    _, _, dg3_tri_up = _get_gravity_triaxial(10, 0, 0, 3, 2, 1, density=1000)
    _, _, dg3_tri_down = _get_gravity_triaxial(-10, 0, 0, 3, 2, 1, density=1000)

    _, _, dg3_obl_up = _get_gravity_oblate(10, 0, 0, 1, 3, 3, density=1000)
    _, _, dg3_obl_down = _get_gravity_oblate(-10, 0, 0, 1, 3, 3, density=1000)

    _, _, dg3_pro_up = _get_gravity_prolate(10, 0, 0, 3, 2, 2, density=1000)
    _, _, dg3_pro_down = _get_gravity_prolate(-10, 0, 0, 3, 2, 2, density=1000)

    np.testing.assert_allclose(np.abs(dg3_tri_down), np.abs(dg3_tri_up))
    np.testing.assert_allclose(np.abs(dg3_pro_down), np.abs(dg3_pro_up))
    np.testing.assert_allclose(np.abs(dg3_obl_down), np.abs(dg3_obl_up))


def test_symmetry_at_constant_radius():
    """

    Testing the symmetry around the sperhical cross section of prolate and
    oblate ellipsoids (axes where b=c).

    """

    a, b, c = (3, 2, 2)
    d, f, g = (2, 3, 3)
    R = 5
    e = 0

    theta = np.linspace(0, 2 * np.pi, 20)
    n = R * np.cos(theta)
    u = R * np.sin(theta)

    _, ogn, ogu = _get_gravity_oblate(e, n, u, d, f, g, density=1000)

    _, pgn, pgu = _get_gravity_prolate(e, n, u, a, b, c, density=1000)

    for i in range(19):
        np.testing.assert_allclose(
            np.sqrt(ogn[i] ** 2 + ogu[i] ** 2),
            np.sqrt(ogn[i + 1] ** 2 + ogu[i + 1] ** 2),
        )
        np.testing.assert_allclose(
            np.sqrt(pgn[i] ** 2 + pgu[i] ** 2),
            np.sqrt(pgn[i + 1] ** 2 + pgu[i + 1] ** 2),
        )


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
