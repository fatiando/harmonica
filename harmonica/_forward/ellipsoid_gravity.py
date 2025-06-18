"""
Forward modelling of a gravity anomaly produced due to an ellipsoidal body.
"""

import numpy as np
from scipy.constants import gravitational_constant as G
from scipy.special import ellipeinc, ellipkinc

from .utils_ellipsoids import _calculate_lambda, _get_V_as_Euler


def ellipsoid_gravity(coordinates, ellipsoids, density, field="g"):
    """
    Compute the three gravity components for an ellipsoidal body at specified o
    bservation locations.

    - Unpacks ellipsoid instance parameters (a, b, c, yaw, pitch, roll, origin)
    - Constructs Euler rotation matrix
    - Rotates observation points (e, n, u) into local ellipsoid system
      (x, y, z)
    - Computes gravity components in local coordinate system
    - Projects these gravity components back into the original coordinate
      system (e, n, u)
    - Returns the gravity field components as arrays

    Parameters
    ----------

    Coordinates: tuple of easting (e), northing (n), upward (u) points
        e : ndarray
            Easting coordinates, in the form:
                - A scalar value (float or int)
                - A 1D array of shape (N,)
                - A 2D array (meshgrid) of shape (M, N)

        n : ndarray
            Northing coordinates, same shape and rules as 'e'.

        u : ndarray
            Upward coordinates, e.g. the surface height desired to compute the
            gravity value. Same shape and rules as 'e'.

    ellipsoid* : value, or list of values
        instance(s) of TriaxialEllipsoid, ProlateEllipsoid,
                 OblateEllipsoid
        Geometric description of the ellipsoid:
            - Semiaxes : a, b, c**
            - Orientation : yaw, pitch, roll**
            - Origin : centre point (x, y, z)

    density***: float or list
        The uniform density of the ellipsoid in kg/m^3, or an array of
        densities for multiple ellipsoid cases, with the same size as
        'ellipsoids'.

    field : (optional) str, one of either "e", "n", "u".
        if no input is given, the function will return all three components of
        gravity.

    Returns
    -------

    ge: ndarray
        Easting component of the gravity field.

    gn ndarray
        Northing component of the gravity field.

    gu: ndarray
        Upward component of the gravity field.

    NOTES

    -----
    * : ellipsoid may be defined using one of the three provided classes:
        TriaxialEllipsoid where a > b > c, OblateEllipsoid where a < b = c,
        and ProlateEllipsoid where a > b = c.
    ** : the c value and roll angle are only explicitly defined for trixial
        ellipsoids by definition of the ellipsoid. Otherwise, ProlateEllipsoid
        and OblateEllipsoid take b = c and roll = 0.
    ***: Must be an array even if multiple ellipsoids are passes, EVEN IF
        the ellipsoids have the same desnity.
    Input arrays must match in shape.

    References
    ----------
    Clark, S. A., et al. (1986), "Magnetic and gravity anomalies of a trixial
    ellipsoid"
    Takenhasi, Y., et al. (2018), "Magentic modelling of ellipsoidal bodies"

    For derivations of the equations, and methods used in this code.
    """

    # unpack coordinates and create array to hold final g values
    e, n, u = coordinates[0], coordinates[1], coordinates[2]
    cast = np.broadcast(e, n, u)
    ge, gn, gu = np.zeros(e.shape), np.zeros(e.shape), np.zeros(e.shape)

    # deal with the case of a single ellipsoid being passed
    if type(ellipsoids) is not list:
        ellipsoids = [ellipsoids]
        density = [density]

    # case of multiple ellipsoids but only one density value
    if type(ellipsoids) is list and type(density) is not list:
        raise ValueError(
            "Ellipsoids is a list, but density is not."
            "Perhaps multiple arguments were given for ellipsoids"
            " but only one value was given for density?"
        )

    # case of passing density as a list of differing size to ellipsoids
    if len(ellipsoids) != len(density):
        raise ValueError(
            f"{len(ellipsoids)} arguments were given for ellipsoids"
            f" but {len(density)} values were given for density."
        )

    for index, ellipsoid in enumerate(ellipsoids):
        # unpack instances
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        ox, oy, oz = ellipsoid.centre

        # preserve ellipsoid shape, translate origin of ellipsoid
        cast = np.broadcast(e, n, u)
        obs_points = np.vstack(((e - ox).ravel(),
                                (n - oy).ravel(), (u - oz).ravel()))

        # create rotation matrix
        R = _get_V_as_Euler(yaw, pitch, roll)

        # rotate observation points
        rotated_points = R.T @ obs_points
        x, y, z = tuple(c.reshape(cast.shape) for c in rotated_points)

        # create boolean for internal vs external field points
        internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1

        # calculate gravity component for the rotated points
        gx, gy, gz = _get_gravity_array(internal_mask, a, b, c,
                                        x, y, z, density[index])
        gravity = np.vstack((gx.ravel(), gy.ravel(), gz.ravel()))

        # project onto upward unit vector, axis U
        g_projected = R @ gravity
        ge_i, gn_i, gu_i = tuple(c.reshape(cast.shape) for c in g_projected)

        # sum contributions from each ellipsoid
        ge += ge_i
        gn += gn_i
        gu += gu_i

    if field == "e":
        return ge
    elif field == "n":
        return gn
    elif field == "u":
        return gu

    return ge, gn, gu


def _get_ABC(a, b, c, lmbda):
    """
    Compute the A(λ), B(λ), and C(λ) functions using elliptic integrals, as
    required for potiential field calculations of ellipsoidal bodies.

    Parameters
    ----------
    x, y, z : array_like or float
        Cartesian observation coordinates. Each can be a scalar, 1D array, or
        2D array, depending on the evaluation grid.

    a, b, c : float
        Semiaxis lengths of the ellipsoid (a ≥ b ≥ c). These must conform to
        the type of ellipsoid used (e.g., triaxial, prolate, oblate).

    lmbda : array_like or float
        The λ (lambda) parameter(s) associated with the confocal ellipsoidal
        coordinate surfaces. Can be a scalar or an array matching the shape of
        the observation points.

    Returns
    -------
    A_lmbda : ndarray
        A(λ) function values at each observation point.

    B_lmbda : ndarray
        B(λ) function values at each observation point.

    C_lmbda : ndarray
        C(λ) function values at each observation point.
    """

    # compute the k and theta terms for the elliptic integrals
    k = np.sqrt((a**2 - b**2) / (a**2 - c**2))
    theta_prime = np.arcsin(np.sqrt((a**2 - c**2) / (a**2 + lmbda)))

    # compute terms associated with A(lambda)
    A_coeff = 2 / ((a**2 - b**2) * np.sqrt(a**2 - c**2))
    A_elliptic_integral = ellipkinc(theta_prime, k) - ellipeinc(theta_prime, k)
    A_lmbda = A_coeff * A_elliptic_integral

    # compute terms associated with B(lambda)
    B_coeff = (2 * np.sqrt(a**2 - c**2)) / ((a**2 - b**2) * (b**2 - c**2))
    B_fk_coeff = (b**2 - c**2) / (a**2 - c**2)
    B_fk_subtracted_term = (
        k**2 * np.sin(theta_prime) * np.cos(theta_prime)
    ) / np.sqrt(1 - k**2 * np.sin(theta_prime) ** 2)
    B_lmbda = B_coeff * (
        ellipeinc(theta_prime, k)
        - B_fk_coeff * ellipkinc(theta_prime, k)
        - B_fk_subtracted_term
    )

    # compute terms associated with C(lambda)
    C_coeff = 2 / ((b**2 - c**2) * np.sqrt(a**2 - c**2))
    C_ek_subtracted_term = (
        np.sin(theta_prime) * np.sqrt(1 - k**2 * np.sin(theta_prime) ** 2)
    ) / np.cos(
        theta_prime
    )  # check the brackets here ??
    C_lmbda = C_coeff * (C_ek_subtracted_term - ellipeinc(theta_prime, k))

    return A_lmbda, B_lmbda, C_lmbda


def _get_internal_g(x, y, z, a, b, c, density):
    """
    Calculate the gravitational field inside a homogeneous ellipsoid.

    Parameters
    ----------
    x, y, z : array or float
        Observation coordinates. Can be scalars, 1D arrays, or 2D arrays.

    a, b, c : float
        Semiaxis lengths of the ellipsoid. Must be consistent with the
        ellipsoid type used.

    density : float
        Uniform density of the ellipsoid (kg/m³).

    Returns
    -------
    g_int_x : ndarray
        x-component of the internal gravitational field.

    g_int_y : ndarray
        y-component of the internal gravitational field.

    g_int_z : ndarray
        z-component of the internal gravitational field.
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
        raise ValueError(
            f"Invalid ellipsoid axis lengths for oblate ellipsoid:"
            f"expected a < b = c but got a = {a}, b = {b}, c = {c}"
        )

    # compute the coefficient of the three delta_g equations
    numerator = np.pi * a * b**2 * G * density
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
        raise ValueError(
            "Invalid ellipsoid axis lengths for prolate ellipsoid: expected"
            f" a > b = c but got a = {a}, b = {b}, c = {c}"
        )

    # compute the coefficient of the three delta_g equations
    numerator = np.pi * a * b**2 * G * density
    denominator = (a**2 - b**2) ** 1.5
    co_eff1 = numerator / denominator

    # compute repeated log_e term
    log_term = np.log(
        ((a**2 - b**2) ** 0.5 + (a**2 + lmbda) ** 0.5) /
        ((b**2 + lmbda) ** 0.5)
    )

    # compute repeated f_2 second term
    f_2_term_2 = (((a**2 - b**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda)

    # compile terms
    dg1 = 4 * co_eff1 * x * (((a**2 - b**2) /
                              (a**2 + lmbda)) ** 0.5 - log_term)
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
    # call and use calc_lambda abd get_ABC functions
    # account for the internal case where lmbda=0
    if lmbda is None:
        lmbda = _calculate_lambda(x, y, z, a, b, c)

    A_lmbda, B_lmbda, C_lmbda = _get_ABC(a, b, c, lmbda)

    # check the function is used for the correct type of ellipsoid
    if not (a > b > c):
        raise ValueError(
            f"Invalid ellipsoid axis lengths for triaxial ellipsoid:"
            f"expected a > b > c but got a = {a}, b = {b}, c = {c}"
        )

    # compute the coefficient of the three delta_g equations
    co_eff = -2 * np.pi * a * b * c * G * density

    # compile all terms
    dg1 = co_eff * x * A_lmbda
    dg2 = co_eff * y * B_lmbda
    dg3 = co_eff * z * C_lmbda

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
        x[~internal_mask], y[~internal_mask],
        z[~internal_mask], a, b, c, density
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
