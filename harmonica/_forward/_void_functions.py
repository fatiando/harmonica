from .utils import _global_to_local

import verde as vd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

##############################################################################

# following functions are not necessary - tested implementations of the paper


def structural_angles_to_abg(strike, dip, rake):  # we can decide if this is needed
    """
    Takes structural (geological) angles of strike, dip, rake and converts them
    into alpha, beta, gamma angles.

    parameters
    ----------
    strike (float):
    dip (0<= dip <= 90)(float):
    rake (float):


    returns
    -------
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate
                                    axis and vertical plane containing major axis
    """

    alpha = strike - np.arcos(
        np.cos(rake) / np.sqrt(1 - np.sin(dip) ** 2 * np.sin(rake) ** 2)
    )

    beta = np.arcsin(np.sin(dip) * np.sin(rake))

    gamma = -np.arctan(np.cot(dip) * np.sec(rake))

    return alpha, beta, gamma


def get_body_rotation_sdr(strike, dip, rake):
    """
    Creates the unit vectors of the body (local) coordinate axis from the
    global coordinate axis, using STRIKE, DIP and RAKE

    parameters
    ----------
    strike (float):
    dip (0<= dip <= 90)(float):
    rake (float):


    Returns
    -------
    V:  v1 [l1, m1, n1] (vector)
        v2 [l2, m2, n2] (vector)
        v3 [l3, m3, n3] (vector)

    """

    if not (0 <= strike <= 360):
        raise ValueError(
            "Invalid value for strike." f"Expected 0<=strike<=360, got {strike}."
        )

    if not (0 <= dip <= 90):
        raise ValueError("Invalid value for dip." f"Expected 0<=dip<=90, got {dip}.")

    if not (0 <= rake <= 180):
        raise ValueError(
            "Invalid value for rake." f"Expected 0<=rake<=180, got {rake}."
        )

    if 0 <= rake <= 90:
        sign = 1
    else:
        sign = -1

    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)

    v1 = [l1, m1, n1] = (
        -np.cos(strike) * np.cos(rake) - np.sin(strike) * np.cos(dip) * np.sin(rake),
        -np.sin(strike) * np.cos(rake) + np.cos(strike) * np.cos(dip) * np.sin(rake),
        -np.sin(dip) * np.sin(rake),
    )
    v2 = [l2, m2, n2] = sign * np.array(
        (
            np.cos(strike) * np.sin(rake) - np.sin(strike) * np.cos(dip) * np.cos(rake),
            np.sin(strike) * np.sin(rake) + np.cos(strike) * np.cos(dip) * np.cos(rake),
            -np.sin(dip) * np.cos(rake),
        )
    )
    v3 = [l3, m3, n3] = sign * np.array(
        (np.sin(strike) * np.sin(dip), -np.cos(strike) * np.sin(dip), -np.sin(dip))
    )

    V = [v1, v2, v3]

    return V


# V1 = get_body_rotation_sdr(0, 90, 0)
# V2 = get_body_rotation_sdr(0, 0, 0)
# print(V1)
# print(V2)


def get_body_rotation_abg(alpha, beta, gamma):  # in degrees
    """
    Creates the unit vectors of the body (local) coordinate axis from the
    global coordinate axis using ALPHA, BETA and GAMMA.

    parameters
    ----------
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate
                                    axis and vertical plane containing major axis

    returns
    -------
    # not sure how is best to lay this out
    V: v1 [l1, m1, n1] (vector)
       v2 [l2, m2, n2] (vector)
       v3 [l3, m3, n3] (vector) :


    """

    # check inputs are valid
    if not (0 <= alpha <= 360):
        raise ValueError(
            "Invalid value for alpha." f"Expected 0<=alpha<=360, got {alpha}."
        )

    if not (0 <= beta <= 90):
        raise ValueError("Invalid value for beta." f"Expected 0<=beta<=90, got {beta}.")

    if not (-90 <= gamma <= 90):
        raise ValueError(
            "Invalid value for gamma." f"Expected -90<=gamma<=90, got {gamma}."
        )

    # convert to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    v1 = [l1, m1, n1] = (
        -np.cos(alpha) * np.cos(beta),
        -np.sin(alpha) * np.cos(beta),
        -np.sin(beta),
    )

    v2 = [l2, m2, n2] = (
        np.cos(alpha) * np.cos(gamma) * np.sin(beta) + np.sin(alpha) * np.sin(gamma),
        np.sin(alpha) * np.cos(gamma) * np.sin(beta) - np.cos(alpha) * np.sin(gamma),
        -np.cos(gamma) * np.cos(beta),
    )

    v3 = [l3, m3, n3] = (
        np.sin(alpha) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma) * np.sin(beta),
        -np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.sin(beta) * np.sin(gamma),
        np.sin(gamma) * np.cos(beta),
    )

    V = [v1, v2, v3]
    # L = (l1, l2, l3)
    # M = (m1, m2, m3)
    # N = (n1, n2, n3)

    return V


# plot the rotation vector as a rotation of the surface.
def plot_axis_rotation(northing, easting, extra_coords, depth, V):
    """
    Plots the plane of rotation of the ellipsoid, as a rotation of the 'surface'.

    Parameters
    ----------
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate
                                    axis and vertical plane containing major axis

    northing, easting, extra_coords (arrays): observation plane to be converted into
    local coordinates. NOTE: 'extra_coords' as given in vd.grid_coordinates refers
    to the height of the plane above the surface.
    depth (float): the depth of the body below the surface.

    Returns
    -------
    None. Produces 3D plot of the surfaces.


    """

    # get local coordinates via rotation
    local_coords = _global_to_local(northing, easting, extra_coords, depth, V)

    # plot both original surface and rotated plane
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot_surface(northing, easting, extra_coords, cmap=cm.jet)
    ax.plot_surface(local_coords[0], local_coords[1], local_coords[2], cmap=cm.jet)

    return


# x1, y1, z1 = generate_basic_ellipsoid()


def _get_ellipsoid_mass(a, b, c, density):
    """
    Get mass of ellipsoid from volume,
    In order to compare to point mass (spherical) source.

    Parameters
    ----------
    a, b, c (m) = ellipsoid semiaxes
    density (kg/m^3) = uniform density of the ellipsoid

    Returns
    -------
    mass of the ellpsoid (kg)

    """
    volume = 4 / 3 * np.pi * a * b * c

    return density * volume


def _get_coords_and_mask(region, spacing, extra_coords, a, b, c, topo_h=None):
    """
    Return the  coordinates and mask which separates points
    within the given ellipsoid and on or outside
    of the given ellipsoid.

    Parameters
    ----------
    region (list)(W, E, S, N): end points of the coordinate grid
    spacing (float): separation between the points (default = 1)
    extra_coords (float or list): surfaces of constant height to test (default = 0)
    a, b, c (float): semiaxes of the ellipsoid

    Returns
    -------
    x, y, z (arrays): 2D coordinate arrays for grid
    internal (array): mask for the internal points of the ellipsoid

    NOTES:
    Consider making it possible to pass a varying array as a set of z coords.
    """
    if topo_h == None:
        e, n, u = vd.grid_coordinates(
            region=region, spacing=spacing, extra_coords=extra_coords
        )

    else:
        e, n = vd.grid_coordinates(region=region, spacing=spacing)
        u = topo_h * np.exp(-(e**2) / (np.max(e) ** 2) - n**2 / (np.max(n) ** 2))

    internal = (e**2) / (a**2) + (n**2) / (b**2) + (u**2) / (c**2) < 1

    return e, n, u, internal
