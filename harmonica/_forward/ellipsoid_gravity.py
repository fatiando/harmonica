# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling of a gravity anomaly produced due to an ellipsoidal body.
"""

from collections.abc import Iterable

import numpy as np
from choclo.constants import GRAVITATIONAL_CONST
from scipy.constants import gravitational_constant as g
from scipy.special import ellipeinc, ellipkinc

from .utils_ellipsoids import (
    _calculate_lambda,
    _get_v_as_euler,
    get_elliptical_integrals,
)


def ellipsoid_gravity(coordinates, ellipsoids, density, field="g"):
    """
    Forward model gravity fields of ellipsoids.

    Compute the gravity acceleration components for an ellipsoidal body at specified
    observation points.

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    ellipsoid : ellipsoid or list of ellipsoids
        Ellipsoidal body represented by an instance of
        :class:`harmonica.TriaxialEllipsoid`, :class:`harmonica.ProlateEllipsoid`, or
        :class:`harmonica.OblateEllipsoid`, or a list of them.
    density : float, list of floats or array
        List or array containing the density of each ellipsoid in kg/m^3.
    field : {"g", "e", "n", "u"}, optional
        Desired field that want to be computed.
        If "e", "n", or "u" the function will return the easting, northing or upward
        gravity acceleration component, respectively.
        If "g", the function will return a tuple with the three components.
        Default to "g".

    Returns
    -------
    ge, gn, gu: arrays
        Easting, northing and upward component of the gravity acceleration.
        Or a single one if ``field`` is "e", "n" or "u".

    References
    ----------
    Clark, S. A., et al. (1986), "Magnetic and gravity anomalies of a triaxial
    ellipsoid"
    Takahashi, Y., et al. (2018), "Magnetic modelling of ellipsoidal bodies"

    For derivations of the equations, and methods used in this code.
    """
    # Cache broadcast of coordinates
    cast = np.broadcast(*coordinates)

    # Ravel coordinates into 1d arrays
    easting, northing, upward = tuple(c.ravel() for c in coordinates)

    # Allocate arrays
    ge, gn, gu = tuple(np.zeros(easting.size) for _ in range(3))

    # deal with the case of a single ellipsoid being passed
    if not isinstance(ellipsoids, Iterable):
        ellipsoids = [ellipsoids]
    if not isinstance(density, Iterable):
        density = [density]

    for ellipsoid, rho in zip(ellipsoids, density, strict=True):
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        origin_e, origin_n, origin_u = ellipsoid.centre

        # Translate observation points to coordinate system in center of the ellipsoid
        obs_points = np.vstack(
            (easting - origin_e, northing - origin_n, upward - origin_u)
        )

        # create rotation matrix
        r = _get_v_as_euler(yaw, pitch, roll)

        # rotate observation points
        rotated_points = r.T @ obs_points
        x, y, z = tuple(c for c in rotated_points)

        # calculate gravity component for the rotated points
        gx, gy, gz = _compute_gravity_ellipsoid(x, y, z, a, b, c, rho)
        gravity = np.vstack((gx, gy, gz))

        # project onto upward unit vector, axis U
        g_projected = r @ gravity
        ge_i, gn_i, gu_i = tuple(c for c in g_projected)

        # sum contributions from each ellipsoid
        ge += ge_i
        gn += gn_i
        gu += gu_i

    # Reshape gravity arrays
    ge, gn, gu = tuple(g.reshape(cast.shape) for g in (ge, gn, gu))

    return {"e": ge, "n": gn, "u": gu}.get(field, (ge, gn, gu))


def _compute_gravity_ellipsoid(x, y, z, a, b, c, density):
    """
    Compute gravity acceleration for an ellipsoid on a set of observation points.

    The observation points can either be internal or external.

    Parameters
    ----------
    x, y, z : arrays
        Observation coordinates in the local ellipsoid reference frame.
    a, b, c : floats
        Semiaxis lengths of the ellipsoid. Must conform to the constraints of
        the chosen ellipsoid type.
    density : float
        Density of the ellipsoidal body in kg/m³.

    Returns
    -------
    gx, gy, gz : arrays
        Gravity acceleration components in the local coordinate system for the
        ellipsoid. Accelerations are given in SI units (m/s^2).
    """
    # Compute lambda for all observation points
    lmbda = _calculate_lambda(x, y, z, a, b, c)

    # Clip lambda to zero for internal points
    inside = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1
    lmbda[inside] = 0

    # Compute gx, gy, gz
    factor = -2 * np.pi * a * b * c * GRAVITATIONAL_CONST * density
    g1, g2, g3 = get_elliptical_integrals(a, b, c, lmbda)
    gx = factor * x * g1
    gy = factor * y * g2
    gz = factor * z * g3

    return gx, gy, gz


# TODO: Leave a single function for the g values, maybe move it to
# `utils_ellipsoids.py`. These are going to be used also by the magnetic code. The
# `_get_gravity_oblate`, `get_gravity_prolate` and `_get_gravity_triaxial` implement the
# same code as this one. Optimize them by caching costly functions (arctan, ellipkinc,
# etc).
def _get_g_values(a, b, c, lmbda):
    """
    Compute the gravity values (g) for the three ellipsoid types.

    Parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.
    lmbda : float
        The given lmbda value for the point we are considering.

    Returns
    -------
    gvals (x, y, z) : floats
        The g values for the given ellipsoid type, and given observation point.
    """
    # Triaxial
    if a > b > c:
        int_arcsin = np.sqrt((a**2 - c**2) / (a**2 + lmbda))
        phi = np.arcsin(int_arcsin)
        k = (a**2 - b**2) / (a**2 - c**2)

        ellipk = ellipkinc(phi, k)
        ellipe = ellipeinc(phi, k)

        g1 = (2 / ((a**2 - b**2) * (a**2 - c**2) ** 0.5)) * (ellipk - ellipe)

        g2_multiplier = (2 * np.sqrt(a**2 - c**2)) / ((a**2 - b**2) * (b**2 - c**2))
        g2_elliptics = ellipe - ((b**2 - c**2) / (a**2 - c**2)) * ellipk
        g2_last_term = ((a**2 - b**2) / np.sqrt(a**2 - c**2)) * np.sqrt(
            (c**2 + lmbda) / ((a**2 + lmbda) * (b**2 + lmbda))
        )

        g2 = g2_multiplier * (g2_elliptics - g2_last_term)

        # Term with the E(k, theta) must have a minus sign
        # (the minus sign is missing in Takahashi (2018)).
        g3_term_1 = -(2 / ((b**2 - c**2) * np.sqrt(a**2 - c**2))) * ellipe
        g3_term_2 = (2 / (b**2 - c**2)) * np.sqrt(
            (b**2 + lmbda) / ((a**2 + lmbda) * (c**2 + lmbda))
        )
        g3 = g3_term_1 + g3_term_2

        gvals_x, gvals_y, gvals_z = g1, g2, g3

    # Prolate
    if a > b and b == c:
        e2 = a**2 - b**2
        sqrt_e = np.sqrt(e2)
        sqrt_l1 = np.sqrt(a**2 + lmbda)
        sqrt_l2 = np.sqrt(b**2 + lmbda)

        # Equation (38): g1
        g1 = (2 / (e2 ** (3 / 2))) * (
            np.log((sqrt_e + sqrt_l1) / sqrt_l2) - sqrt_e / sqrt_l1
        )

        # Equation (39): g2 = g3
        g2 = (1 / (e2 ** (3 / 2))) * (
            (sqrt_e * sqrt_l1) / (b**2 + lmbda) - np.log((sqrt_e + sqrt_l1) / sqrt_l2)
        )
        gvals_x, gvals_y, gvals_z = g1, g2, g2

    # Oblate
    if a < b and b == c:
        g1 = (
            2
            / ((b**2 - a**2) ** (3 / 2))
            * (
                (np.sqrt((b**2 - a**2) / (a**2 + lmbda)))
                - np.arctan(np.sqrt((b**2 - a**2) / (a**2 + lmbda)))
            )
        )
        g2 = (
            1
            / ((b**2 - a**2) ** (3 / 2))
            * (
                np.arctan(np.sqrt((b**2 - a**2) / (a**2 + lmbda)))
                - (np.sqrt((b**2 - a**2) * (a**2 + lmbda))) / (b**2 + lmbda)
            )
        )

        gvals_x, gvals_y, gvals_z = g1, g2, g2

    return gvals_x, gvals_y, gvals_z


def _get_internal_g(x, y, z, a, b, c, density):
    """
    Calculate the gravitational field inside a homogeneous ellipsoid.

    Parameters
    ----------
    x, y, z : arrays or floats
        Observation coordinates. Can be scalars, 1D arrays, or 2D arrays.
    a, b, c : floats
        Semiaxis lengths of the ellipsoid. Must be consistent with the
        ellipsoid type used.
    density : float
        Uniform density of the ellipsoid (kg/m³).

    Returns
    -------
    g_int_x, g_int_y, g_int_y : arrays
        x-, y-, and z-components of the internal gravitational field.
    """
    # calculate functions with lambda = 0
    # in the triaxial case
    if b != c:
        g_int_x, g_int_y, g_int_z = _get_gravity_triaxial(
            x, y, z, a, b, c, density, lmbda=0
        )

    # in the prolate case
    elif a > b:
        g_int_x, g_int_y, g_int_z = _get_gravity_prolate(
            x, y, z, a, b, c, density, lmbda=0
        )

    # in the oblate case
    else:
        g_int_x, g_int_y, g_int_z = _get_gravity_oblate(
            x, y, z, a, b, c, density, lmbda=0
        )

    return g_int_x, g_int_y, g_int_z


def _get_gravity_oblate(x, y, z, a, b, c, density, lmbda=None):
    """
    Calculate the components of Δg₁, Δg₂, and Δg₃ for the oblate ellipsoid case
    (a < b = c).

    Δgᵢ represents the components of the gravitational field change along the
    local principal axes of the ellipsoid.

    Parameters
    ----------
    x, y, z : array or float
        Observation coordinates in the local ellipsoid reference frame.
        Can be scalars, 1D arrays, or 2D arrays.

    a, b, c : float
        Semiaxis lengths of the ellipsoid. Must satisfy the condition a < b = c
        for the oblate ellipsoid case.

    density : float
        Density of the ellipsoidal body (in kg/m³).

    lmbda : float or array
        λ values used in the internal potential field solution, i.e. for the
        case where λ = 0 inside the ellipsoid. Otherwise is 'None' and a
        ppropriate λ values are computed internally based on the observation
        coordinates.


    Returns
    -------
    g1 : ndarray
        Δg₁ component — change in gravity along the local x-axis.

    g2 : ndarray
        Δg₂ component — change in gravity along the local y-axis.

    g3 : ndarray
        Δg₃ component — change in gravity along the local z-axis.
    """
    # call and use lambda function
    if lmbda is None:
        lmbda = _calculate_lambda(x, y, z, a, b, c)

    # check the function is used for the correct type of ellipsoid
    if not (a < b and b == c):
        msg = (
            f"Invalid ellipsoid axis lengths for oblate ellipsoid:"
            f"expected a < b = c but got a = {a}, b = {b}, c = {c}"
        )
        raise ValueError(msg)

    # compute the coefficient of the three delta_g equations
    numerator = np.pi * a * b**2 * g * density
    denominator = (b**2 - a**2) ** 1.5
    co_eff1 = numerator / denominator

    # compute repeated arctan term
    arc_tan_term = np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)

    # compute the terms within the brackets for delta_g 1,2,3
    bracket_term_g1 = arc_tan_term - ((b**2 - a**2) / (a**2 + lmbda)) ** 0.5

    bracket_term_g2g3 = (
        (((b**2 - a**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda)
    ) - arc_tan_term

    # compile constants, coefficients, bracket terms to calculate final
    # value of the delta_g terms
    g1 = 4 * co_eff1 * x * bracket_term_g1
    g2 = 2 * co_eff1 * y * bracket_term_g2g3
    g3 = 2 * co_eff1 * z * bracket_term_g2g3

    return g1, g2, g3


def _get_gravity_prolate(x, y, z, a, b, c, density, lmbda=None):
    """

    Calculate the components of Δg₁, Δg₂, and Δg₃ for the prolate ellipsoid
    case (a > b = c).

    Δgᵢ represents the components of the gravitational field change along the
    local principal axes of the ellipsoid.

    Parameters
    ----------
    x, y, z : array or float
        Observation coordinates in the local ellipsoid reference frame.
        Can be scalars, 1D arrays, or 2D arrays.

    a, b, c : float
        Semiaxis lengths of the ellipsoid. Must satisfy the condition a > b = c
        for the prolate ellipsoid case.

    density : float
        Density of the ellipsoidal body (in kg/m³).

    lmbda : float or array
        λ values used in the internal potential field solution, i.e. for the
        case where λ = 0 inside the ellipsoid. Otherwise is 'None' and
        appropriate λ values are computed internally based on the observation
        coordinates.


    Returns
    -------
    g1 : ndarray
        Δg₁ component — change in gravity along the local x-axis.

    g2 : ndarray
        Δg₂ component — change in gravity along the local y-axis.

    g3 : ndarray
        Δg₃ component — change in gravity along the local z-axis.
    """
    # call and use lambda function
    if lmbda is None:
        lmbda = _calculate_lambda(x, y, z, a, b, c)

    # check the function is used for the correct type of ellipsoid
    if not (a > b and b == c):
        msg = (
            "Invalid ellipsoid axis lengths for prolate ellipsoid: expected"
            f" a > b = c but got a = {a}, b = {b}, c = {c}"
        )
        raise ValueError(msg)

    # compute the coefficient of the three delta_g equations
    numerator = np.pi * a * b**2 * g * density
    denominator = (a**2 - b**2) ** 1.5
    co_eff1 = numerator / denominator

    # compute repeated log_e term
    log_term = np.log(
        ((a**2 - b**2) ** 0.5 + (a**2 + lmbda) ** 0.5) / ((b**2 + lmbda) ** 0.5)
    )

    # compute repeated f_2 second term
    f_2_term_2 = (((a**2 - b**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda)

    # compile terms
    dg1 = 4 * co_eff1 * x * (((a**2 - b**2) / (a**2 + lmbda)) ** 0.5 - log_term)
    dg2 = 2 * co_eff1 * y * (log_term - f_2_term_2)
    dg3 = 2 * co_eff1 * z * (log_term - f_2_term_2)

    return dg1, dg2, dg3


def _get_gravity_triaxial(
    x, y, z, a, b, c, density, lmbda=None
):  # takes semiaxes, lambda value, density
    """
    Calculate the components of Δg₁, Δg₂, and Δg₃ for the triaxial ellipsoid
    case (a > b > c).

    Δgᵢ represents the components of the gravitational field change along the
    local principal axes of the ellipsoid.

    Parameters
    ----------
    x, y, z : array or float
        Observation coordinates in the local ellipsoid reference frame.
        Can be scalars, 1D arrays, or 2D arrays.

    a, b, c : float
        Semiaxis lengths of the ellipsoid. Must satisfy the condition a > b > c
        for the triaxial ellipsoid case.

    density : float
        Density of the ellipsoidal body (in kg/m³).

    lmbda : float or array
        λ values used in the internal potential field solution, i.e. for the
        case where λ = 0 inside the ellipsoid. Otherwise is 'None' and
        appropriate λ values are computed internally based on the observation
        coordinates.


    Returns
    -------
    g1 : ndarray
        Δg₁ component — change in gravity along the local x-axis.

    g2 : ndarray
        Δg₂ component — change in gravity along the local y-axis.

    g3 : ndarray
        Δg₃ component — change in gravity along the local z-axis.
    """
    # call and use calc_lambda abd get_abc functions
    # account for the internal case where lmbda=0
    if lmbda is None:
        lmbda = _calculate_lambda(x, y, z, a, b, c)

    a_lmbda, b_lmbda, c_lmbda = _get_g_values(a, b, c, lmbda)

    # check the function is used for the correct type of ellipsoid
    if not (a > b > c):
        msg = (
            f"Invalid ellipsoid axis lengths for triaxial ellipsoid:"
            f"expected a > b > c but got a = {a}, b = {b}, c = {c}"
        )
        raise ValueError(msg)

    # compute the coefficient of the three delta_g equations
    co_eff = -2 * np.pi * a * b * c * g * density

    # compile all terms
    dg1 = co_eff * x * a_lmbda
    dg2 = co_eff * y * b_lmbda
    dg3 = co_eff * z * c_lmbda

    return dg1, dg2, dg3


def _get_gravity_array(internal_mask, a, b, c, x, y, z, density):
    """ "
    Compute the total gravitational effect of an ellipsoidal body at given
    observation points.

    Combines of external and internal g calculations for a given ellipsoid.

    Parameters
    ----------
    x, y, z : array or float
        Observation coordinates in the local ellipsoid reference frame.
        Can be scalars, 1D arrays, or 2D arrays.

    a, b, c : float
        Semiaxis lengths of the ellipsoid. Must conform to the constraints of
        the chosen ellipsoid type.

    density : float
        Density of the ellipsoidal body in kg/m³.

    internal_mask : array_like of bool
        Boolean mask indicating which observation points lie inside the
        ellipsoid (`True` for inside, `False` for outside).

    Returns
    -------
    xresults : ndarray
        Gravitational field component in the local x-direction.

    yresults : ndarray
        Gravitational field component in the local y-direction.

    zresults : ndarray
        Gravitational field component in the local z-direction.

    """
    # select function to use based on ellipsoid parameters
    if a > b > c:
        func = _get_gravity_triaxial
    elif a > b and b == c:
        func = _get_gravity_prolate
    elif a < b and b == c:
        func = _get_gravity_oblate

    # create array to hold values
    xresults = np.zeros(x.shape)
    yresults = np.zeros(y.shape)
    zresults = np.zeros(z.shape)

    # call functions to produce g values, external and internal
    g_ext_x, g_ext_y, g_ext_z = func(
        x[~internal_mask],
        y[~internal_mask],
        z[~internal_mask],
        a,
        b,
        c,
        density,
    )
    g_int_x, g_int_y, g_int_z = _get_internal_g(
        x[internal_mask], y[internal_mask], z[internal_mask], a, b, c, density
    )

    # assign external and internal values to the arrays created
    xresults[internal_mask] = g_int_x
    xresults[~internal_mask] = g_ext_x

    yresults[internal_mask] = g_int_y
    yresults[~internal_mask] = g_ext_y

    zresults[internal_mask] = g_int_z
    zresults[~internal_mask] = g_ext_z

    return xresults, yresults, zresults
