import numpy as np
from scipy.spatial.transform import Rotation as R


def _calculate_lambda(x, y, z, a, b, c):
    """
    Calculate the value of lambda, the parameter defining surfaces in a
    confocal family of ellipsoids (i.e., the inflation/deflation parameter),
    for a given ellipsoid and observation point.

    Parameters
    ----------
    x : float or array
        X-coordinate(s) of the observation point(s) in the local coordinate
        system.
    y : float or array
        Y-coordinate(s) of the observation point(s).
    z : float or array
        Z-coordinate(s) of the observation point(s).
    a : float
        Semi-major axis of the ellipsoid along the x-direction.
    b : float
        Semi-major axis of the ellipsoid along the y-direction.
    c : float
        Semi-major axis of the ellipsoid along the z-direction.

    Returns
    -------
    lmbda : float or array-like
        The computed value(s) of the lambda parameter.

    """
    if not (np.any(np.abs(x) >= a) or np.any(np.abs(y) >= b)
            or np.any(np.abs(z) >= c)):
        raise ValueError(
            "Arrays x, y, z should contain points which lie outside"
            " of the surface defined by a, b, c"
        )

    # compute lambda
    p_0 = (
        a**2 * b**2 * c**2
        - b**2 * c**2 * x**2
        - c**2 * a**2 * y**2
        - a**2 * b**2 * z**2
    )
    p_1 = (
        a**2 * b**2
        + b**2 * c**2
        + c**2 * a**2
        - (b**2 + c**2) * x**2
        - (c**2 + a**2) * y**2
        - (a**2 + b**2) * z**2
    )
    p_2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2

    p = p_1 - (p_2**2) / 3

    q = p_0 - ((p_1 * p_2) / 3) + 2 * (p_2 / 3) ** 3

    theta_internal = -q / (2 * np.sqrt((-p / 3) ** 3))

    # clip to remove floating point precision errors (as per testing)
    theta_internal_1 = np.clip(theta_internal, -1.0, 1.0)

    theta = np.arccos(theta_internal_1)

    lmbda = 2 * np.sqrt((-p / 3)) * np.cos(theta / 3) - p_2 / 3

    return lmbda


def _get_V_as_Euler(yaw, pitch, roll):
    """
    Generate a rotation matrix (V) from Tait-Bryan angles: yaw, pitch,
    and roll.

    Parameters
    ----------
    yaw : float
        Rotation about the vertical (z) axis, in degrees.
    pitch : float
        Rotation about the northing (y) axis, in degrees.
    roll : float
        Rotation about the easting (x) axis, in degrees.

    These rotations are applied in the following order order as above, (zyx).

    Returns
    -------
    V : ndarray of shape (3, 3)
        Rotation matrix that transforms coordinates from the local
        ellipsoid-aligned
        frame to the global coordinate system.

    Notes
    -----
    All angles must be given in degrees.

    """

    # using scipy rotation package
    # this produces the local to global rotation matrix (or what would be
    # defined
    # as R.T from global to local)
    r = R.from_euler("zyx", [yaw, -pitch, roll], degrees=True)
    V = r.as_matrix()

    return V


def _global_to_local(northing, easting, extra_coords, depth, V):
    """
    Convert observation points from global coordinates (Northing, Easting,
                                                        Height)
    to local ellipsoid-aligned coordinates (x, y, z).

    Parameters
    ----------
    northing : array_like
        Northing (Y) coordinates in the global system.

    easting : array_like
        Easting (X) coordinates in the global system.

    extra_coords : array_like
        Height or vertical offset above the surface (commonly from
                                                     `vd.grid_coordinates`).

    depth : float
        Depth of the ellipsoid’s center below the surface (positive downward).

    V : ndarray of shape (3, 3)
        Rotation matrix used to transform from global to local coordinates.

    Returns
    -------
    x, y, z : ndarray
        Coordinates of the observation points in the local ellipsoid-aligned
        frame.

    Notes
    -----
    Needs to handle translation component.

    """

    x = np.ones(northing.shape)
    y = np.ones(northing.shape)
    z = np.ones(northing.shape)
    local_coords = [x, y, z]

    # calculate local_coords for each x, y, z point
    for i in range(len(local_coords)):
        local_coords[i] = (
            northing * V[i][0] + easting * V[i][1] -
            (depth - extra_coords) * V[i][2]
        )

    return local_coords


def _generate_basic_ellipsoid(a, b, c):
    """
    Generate the surface of an ellipsoid using spherical angles for 3D
    plotting.
    This function is seperate from gravity calculations and is purely for
    visualisation of 3D ellipsoids.

    Parameters
    ----------
    a, b, c : float
        Semiaxis lengths of the ellipsoid along the x, y, and z axes,
        respectively.

    Returns
    -------
    x1, y1, z1 : ndarray
        Arrays representing the ellipsoid surface coordinates in 3D space,
        computed
        from spherical angles. T

    """

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # np.outer is the outer product of the two arrays (ellipsoid surfce)
    x1 = a * np.outer(np.cos(u), np.sin(v))
    y1 = b * np.outer(np.sin(u), np.sin(v))
    z1 = c * np.outer(np.ones_like(u), np.cos(v))

    return x1, y1, z1
